import tensorflow as tf
import math
import numpy as np

class AttRec(object):
    def __init__(self, config):

        self.emb_size = config.embedding_size
        self.item_count = config.item_count
        self.user_count = config.user_count
        self.msl = config.sequence_length
        self.tsl = config.target_length
        self.nsl = config.neg_sample_count
        self.w = config.w
        self.gamma = config.gamma
        self.l2_lambda = config.l2_lambda
        #self.u_init = tf.random_uniform_initializer(minval=-1.0 / self.emb_size,maxval=1.0 / self.emb_size)
        self.u_init = tf.keras.initializers.he_normal()
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_Training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.hist_seq = tf.placeholder(tf.int32, [None, None]) #[B,T]
        self.u_p = tf.placeholder(tf.int32, [None]) #[B]
        self.neg_p = tf.placeholder(tf.int32, [None, None]) #[B,F]
        self.sl = tf.placeholder(tf.int32, [None]) #[B]
        self.next_p = tf.placeholder(tf.int32, [None,None]) #[B]

        self.build_model()

    def build_model(self):
        with tf.variable_scope('AttRec'):
            # Embedding
            self.item_emb = tf.get_variable("item_emb", [self.item_count, self.emb_size],initializer=self.u_init) #[N,e]

            self.item_rep_emb = tf.get_variable("item_rep_emb", [self.item_count, self.emb_size],initializer=self.u_init) #[N,e]
            self.user_rep_emb = tf.get_variable("user_rep_emb", [self.user_count, self.emb_size],initializer=self.u_init) #[N,e]

            value = tf.nn.embedding_lookup(self.item_emb, self.hist_seq,max_norm=1)  # [B,T,e]

            value = self.mask_seq(value,self.sl)
            #Add TimeSignals
            ts = self.make_time_signal(self.emb_size, self.msl) #[msl, e]
            ts = tf.tile(tf.expand_dims(ts, [0]), [tf.shape(self.sl)[0], 1, 1])

            query = tf.add(value, ts) #[B,T,e]
            key = query

            #Self Attention
            output = self.attention_module(query,key,value,self.emb_size)

            # Mean
            div_num = tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, self.emb_size]), tf.float32)
            m = tf.div(tf.reduce_sum(output, axis=1), div_num) #[B,e]
            
            #Look up embedding for targets and negative samples
            u = tf.nn.embedding_lookup(self.user_rep_emb, self.u_p) #[B,e]
            u = tf.nn.l2_normalize(u,-1)
            u = tf.clip_by_norm(u, 1, -1)

            pos_v = tf.nn.embedding_lookup(self.item_rep_emb, self.next_p) #[B,pos,e]
            pos_v = tf.clip_by_norm(pos_v,1,-1)

            neg_v = tf.nn.embedding_lookup(self.item_rep_emb, self.neg_p) #[B,neg,e]
            neg_v = tf.clip_by_norm(neg_v, 1, -1)

            pos_x = tf.nn.embedding_lookup(self.item_emb, self.next_p) #[B,pos,e]
            pos_x = tf.clip_by_norm(pos_x, 1, -1)


            neg_x = tf.nn.embedding_lookup(self.item_emb, self.neg_p)#[B,neg,e]
            neg_x = tf.clip_by_norm(neg_x, 1, -1)

            self.pos_y = pos_y = self.pos_object_function(u, pos_v, m, pos_x, self.w) #[B,tsl]
            self.neg_y = neg_y = self.neg_object_function(u, neg_v, m, neg_x, self.w) #[B,nsl]

            #margin based hinge loss
            self.loss = self.loss_function(self.gamma, pos_y, neg_y)
            self.next_items = m


    def attention_module(self,query,key,value,unit):
        with tf.variable_scope('attention',reuse=True):
            query = tf.layers.dense(query, unit, name='qk_map', activation=tf.nn.relu, use_bias=False,kernel_initializer=self.u_init,
                                    reuse=tf.AUTO_REUSE,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            query = tf.nn.dropout(query, self.keep_prob)

            key = tf.layers.dense(key, self.emb_size, name='qk_map', activation=tf.nn.relu, use_bias=False,kernel_initializer=self.u_init,
                                  reuse=tf.AUTO_REUSE,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            key = tf.nn.dropout(key, self.keep_prob)

            score = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / math.sqrt(self.emb_size)  # [B,T,T]

            #masks the diagonal of the affinity matrix
            a_mask = tf.ones([tf.shape(score)[1], tf.shape(score)[2]])
            a_mask = a_mask - tf.matrix_diag(tf.ones([tf.shape(score)[1]]))
            a_mask = tf.expand_dims(a_mask, [0])
            a_mask = tf.tile(a_mask, [tf.shape(score)[0], 1, 1])
            score *= a_mask
            score = tf.nn.softmax(score, axis=2)
            output = tf.matmul(score, value)
            return output



    def pos_object_function(self, U, V, m, X, w):
        m = tf.tile(tf.expand_dims(m, [1]), [1, self.tsl, 1])
        U = tf.tile(tf.expand_dims(U, [1]), [1, self.tsl, 1])
        return w * tf.reduce_sum(tf.square(U - V), axis=-1) + (1 - w) * tf.reduce_sum(tf.square(m - X), axis=-1)


    def neg_object_function(self, U, V, m, X, w):
        m = tf.tile(tf.expand_dims(m, [1]), [1, self.nsl, 1])
        U = tf.tile(tf.expand_dims(U, [1]), [1, self.nsl, 1])
        return w * tf.reduce_sum(tf.square(U - V), axis=-1) + (1 - w) * tf.reduce_sum(tf.square(m - X), axis=-1)

    def loss_function(self, gamma, pos_y, neg_y):
        pos_y = tf.reshape(tf.tile(tf.expand_dims(pos_y, -1), [1, 1, self.nsl]), [-1, self.tsl * self.nsl])
        neg_y = tf.reshape(tf.tile(neg_y,[1,self.tsl]),[-1,self.tsl * self.nsl])
        loss = tf.reduce_mean(tf.nn.relu(pos_y + gamma - neg_y), axis=-1)
        return loss

    def mask_seq(self, input, input_length):

        mask = tf.sequence_mask(input_length, tf.shape(input)[1], dtype=tf.float32)  # [B,T]
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        mask = tf.tile(mask, [1, 1, tf.shape(input)[2]])  # [B,T,e]
        input *= mask  # [B,T,e]
        return input


    def TE(self, t, i, d):
        if i % 2 == 0:
            return math.sin(t / math.pow(10000, 2 * i / d))
        else:
            return math.cos(t / math.pow(10000, 2 * (i - 1) / d))

    def make_time_signal(self, size, max_timestep):
        te_list = []
        for t in range(max_timestep):
            tmp = []
            for i in range(size):
                tmp.append(self.TE(t, i, size))
            te_list.append(tmp)
        return te_list

    def predict(self,item_list,topk):

        all_idx = tf.convert_to_tensor(item_list,dtype=tf.int32)

        u = tf.nn.embedding_lookup(self.user_rep_emb, self.u_p)
        U = tf.tile(tf.expand_dims(u, [1]), [1, self.item_count, 1])
        m = tf.tile(tf.expand_dims(self.next_items, [1]), [1, self.item_count, 1])

        item_r = tf.nn.embedding_lookup(self.item_rep_emb, all_idx, max_norm=1)
        item_e = tf.nn.embedding_lookup(self.item_emb, all_idx, max_norm=1)

        score = self.w * tf.reduce_sum(tf.square(U - item_r), axis=-1) + (1 - self.w) * tf.reduce_sum(
            tf.square(m - item_e), axis=-1)
        top_k = tf.nn.top_k(-1 * score, k=topk)
        topk_index = top_k.indices

        return topk_index



















