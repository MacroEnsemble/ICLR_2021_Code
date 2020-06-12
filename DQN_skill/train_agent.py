import numpy as np
import tensorflow as tf
import random
import time

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    start = time.time()
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

    end = time.time()
    print("copy model parameters {} seconds.".format(end - start))

# Train agent
def train_agent(
        sess,
        replay_memory,
        q_estimator,
        target_estimator,
        discount_factor,
        NUM_EPOCH,
        BATCH_SIZE,
        NUM_TOTAL_SKILL,
        update_target_estimator_every,
        tf_update_idx,
    ):
    print("Start training agent.")
    print('Replay memory size: {}'.format(len(replay_memory)))
    NUM_BATCH = len(replay_memory) // BATCH_SIZE
    
    # Soft update
    copy_model_parameters(sess, q_estimator, target_estimator)
    
    for e in range(NUM_EPOCH):
        random.shuffle(replay_memory)
        for i in range(NUM_BATCH):
            samples = replay_memory[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            
            # Soft update
            # if (e * NUM_BATCH + i) % update_target_estimator_every == 0:
            #     copy_model_parameters(sess, q_estimator, target_estimator)

            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(BATCH_SIZE), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            abs_loss = q_estimator.get_abs_loss(sess, states_batch, action_batch, targets_batch)

            print("\rUpdate {}/{} @ emsenble {}/{}, loss: {}".format(e * NUM_BATCH + i + 1, NUM_EPOCH * NUM_BATCH,
                                                                    tf_update_idx, NUM_TOTAL_SKILL, loss), end="")
    print("")

    return q_estimator, target_estimator

if __name__ == '__main__':
    pass