from gym_torcs import TorcsEnv
import numpy as np
import argparse
import time
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import time
import tensorflow as tf
from gym_torcs import TorcsEnv
import json
from models import Generator
from models import Posterior
from keras.models import Model
import cv2

from models import TRPOAgent
import matplotlib.pyplot as plt

MAX_STEP_LIMIT = 300
MIN_STEP_LIMIT = 100
PRE_STEP = 100

feat_dim = [7, 13, 1024]
aux_dim = 10
encode_dim = 2
action_dim = 3
img_dim = [50, 50, 3]

code = 1
param_dir = "/home/Apurba/Documents/RL/params/"
preactions_path = "/home/Apurba/Documents/RL/human_0/pre_actions.npz"
i = 229


def get_state(ob, aux_dim, feat_extractor):
    imgN = ob.img
    img = ob.img
    #print("shape is",img.shape)
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    img = cv2.resize(img, (200, 150))
    #print("shape now is",img.shape)
    img = img[40:, :, :]
    x = np.expand_dims(img, axis=0).astype(np.float32)
    x = preprocess_input(x)
    feat = feat_extractor.predict(x)

    aux = np.zeros(aux_dim, dtype=np.float32)
    aux[0] = ob.damage
    aux[1] = ob.speedX
    aux[2] = ob.speedY
    aux[3] = ob.speedZ
    aux[4:7] = ob.pre_action_0
    aux[7:10] = ob.pre_action_1
    return feat, np.expand_dims(aux, axis=0), img

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=config)
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    iterX = []
    distY = []
    encodeAcc = []
    base_model = ResNet50(weights='imagenet', include_top=False)
    feat_extractor = Model(input=base_model.input,
                                output=base_model.get_layer('activation_40').output)
    for iterations in range(38,229,9):
        iterX.append(iterations)
        generator = Generator(sess, feat_dim, aux_dim, encode_dim, action_dim)
        posterior = Posterior(sess, img_dim, aux_dim, action_dim, encode_dim)

        
        paramgen_path  = param_dir + "generator_model_" +str(iterations)+".h5"
        print(paramgen_path)
        #parampost_path = param_dir + "posterior_model_" +str(iterations)+".h5"
        try:
            generator.model.load_weights(paramgen_path)
            #posterior.model.load_weights(parampost_path)
            print("Weights loaded successfully")
        except:
            print("cannot find weight")

        env = TorcsEnv(throttle=True, gear_change=False)

        print("Start driving ...")
        ob = env.reset(relaunch=True)
        feat, aux, img = get_state(ob, aux_dim, feat_extractor)

        encode = np.zeros((1, encode_dim), dtype=np.float32)
        encode[0, code] = 1
        print ("Encode:", encode[0])

        pre_actions = np.load(preactions_path)["actions"]
        for i in xrange(MAX_STEP_LIMIT):
            if i < MIN_STEP_LIMIT:
                action = np.zeros(3, dtype=np.float32)
            elif i < MIN_STEP_LIMIT + PRE_STEP:
                action = pre_actions[i - MIN_STEP_LIMIT]
            else:
                action = generator.model.predict([feat, aux, encode])[0]

            ob, reward, done, _ = env.step(action)
            feat, aux,img = get_state(ob, aux_dim, feat_extractor)

            if (i == PRE_STEP + MIN_STEP_LIMIT):
                print ("Start deciding...")
        
        if done:
                distY.append(ob.distFromStart)
                break
   
        distY.append(ob.distFromStart)
        print ("iteration:",iterations,"Step:", i, "DistFromStart:", ob.distFromStart, \
                "TrackPos:", ob.trackPos, "Damage:", ob.damage.item(), \
                "Action: %.6f %.6f %.6f" % (action[0], action[1], action[2]), \
                "Speed:", ob.speedX * 200)

                
        #encodeAcc.append(prediction)
    
    env.end()
    print("Finish.")

    print(distY)
    
    plt.plot(iterX, distY)
    plt.plot(avg_rewards)
    plt.xlabel('Episode No->', fontweight = 'bold')
    plt.ylabel('Distance from the start->', fontweight = 'bold')
    plt.title('Distance from start over multiple rollouts vs episode no', fontweight = 'bold')
    plt.show()
    plt.savefig("pass_left.pdf",bbox_inches = 'tight')




if __name__ == "__main__":
    main() 
        




