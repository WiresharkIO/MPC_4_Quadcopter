import logging
import time
import sys
from threading import Event
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

def sendControl(current_U):
    sequence_with_LPS(current_U)

def position_callback(timestamp, data, logconf):
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    # Intermediate States to MPC
    # return(x, y, z)
    # print('pos: ({}, {}, {})'.format(x, y, z))

def start_position_printing(scf):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')

    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()

def sequence_with_LPS(sequence):
    DEFAULT_HEIGHT = 0.5
    print('Setting velocity {}'.format(sequence))
    cflib.crtp.init_drivers()
    time.sleep(1)
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            cf = scf.cf
            # DEFINE SEQUENCE
            time.sleep(1)
            for i in range(1):
                mc.start_linear_motion(sequence[0],
                                       sequence[1],
                                       sequence[2],
                                       sequence[3])
                time.sleep(0.2)
                start_position_printing(scf)
        mc.stop()