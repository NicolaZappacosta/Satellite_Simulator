import numpy as np
import pandas as pd
from datetime import datetime
import os

class EOM(object):

    def __init__(self,ID: int = 0, position = [0, 0, 0], velocity = [0, 0, 0], quaternion = [0, 0, 0, 0], angular_rate = [0, 0, 0], time = 0):
        self.object_date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.ID = ID
        self.r_G_I = np.array(position).reshape(3)
        self.v_GI_I = np.array(velocity).reshape(3)
        self.q_BI = np.array(quaternion).reshape(4)
        self.w_BI_B = np.array(angular_rate).reshape(3)
        self.t = np.array(time)
        self.HISTORY = self._updateHISTORY()

    def updateEoM(self, position, velocity, quaternion, angular_rate, time):
        self.r_G_I = position.values[-1][:].reshape(3)
        self.v_GI_I = velocity.values[-1][:].reshape(3)
        self.q_BI = quaternion.values[-1][:].reshape(4)
        self.w_BI_B = angular_rate.values[-1][:].reshape(3)
        self.t = time
        data = self.eom2HISTORY(position, velocity, quaternion, angular_rate, time)
        self.HISTORY = pd.concat([self.HISTORY, data])

    def appendHISTORY(self, data):
        HISTORY = pd.concat([self.HISTORY, data])
        self.HISTORY = HISTORY
        self.r_G_I = np.array([HISTORY[['x1', 'x2', 'x3']].values[-1]]).reshape(3)
        self.v_GI_I = np.array([HISTORY[['v1', 'v2', 'v3']].values[-1]]).reshape(3)
        self.w_BI_B = np.array([HISTORY[['w1', 'w2', 'w3']].values[-1]]).reshape(3)
        self.q_BI = np.array([HISTORY[['q1', 'q2', 'q3', 'q0']].values[-1]]).reshape(4)
        self.t = HISTORY['t'].values[-1]

    def printHISTORY(self):
        print(self.HISTORY)
    def getHISTORY(self):
        return self.HISTORY
    def saveHISTORY(self,path: str = "."):
        #TODO: EXTERNALIZE THE INPUT FOR THE SAVING DIRECTORY.
        #TODO: DATAHANDLING TO BE IMPROVED. THE DATA FROM CAN BE DISCHARGED AND APPENDED.
        #TODO: LOADING HISTORY
        print('-'*20+' Saving History ID: '+str(self.ID)+'-'*20+' \n')
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        RESULTS_DIR = ROOT_DIR+"\Result"

        if not os.path.isdir(RESULTS_DIR):
            print('Result directory not detected. Generating directory...\n')
            os.mkdir(RESULTS_DIR)
            print('Directory generated: '+ RESULTS_DIR)

        SAVE_DIR = ROOT_DIR+"\Result"+path[1:].replace('/','\\')

        if not os.path.isdir(SAVE_DIR):
            print('Save directory not detected. Generating directory...\n')
            os.mkdir(SAVE_DIR)
            print('Directory generated: '+ SAVE_DIR)

        FILE_PATH = SAVE_DIR+"\\ID"+str(self.ID)+".csv"

        if os.path.isfile(FILE_PATH):
            print('Overwriting data...')

        self.HISTORY.to_csv(FILE_PATH)
    ## Private met
    def _updateHISTORY(self):
        if not hasattr(self, 'history'):
            history = self.eom2HISTORY(self.r_G_I, self.v_GI_I, self.q_BI, self.w_BI_B, self.t)
        else:
            history = pd.concat([self.history, self.eom2history(self.r_G_I, self.v_GI_I, self.q_BI, self.w_BI_B, self.t)])

        return history

    @staticmethod
    def eom2HISTORY(position, velocity, quaternion, angular_rate, time):
        #TODO: this is spagetthi code. Change it to be more straightforward.
        data = pd.DataFrame()
        data['t'] = [time]
        data['q1'] = [quaternion[0]]
        data['q2'] = [quaternion[1]]
        data['q3'] = [quaternion[2]]
        data['q0'] = [quaternion[3]]
        data['w1'] = [angular_rate[0]]
        data['w2'] = [angular_rate[1]]
        data['w3'] = [angular_rate[2]]
        data['x1'] = [position[0]]
        data['x2'] = [position[1]]
        data['x3'] = [position[2]]
        data['v1'] = [velocity[0]]
        data['v2'] = [velocity[1]]
        data['v3'] = [velocity[2]]

        return data
