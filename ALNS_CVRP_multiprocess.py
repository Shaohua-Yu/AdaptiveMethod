import time
import csv
import math
import random
import numpy as np
import copy
import sys
import multiprocessing 
import json 

# --- Sol Class Definition ---
class Sol():
    def __init__(self):
        self.node_no_seq = None
        self.obj = float('inf')
        self.route_list = None
        self.route_distance = None

# --- Node Class Definition ---
class Node():
    def __init__(self):
        self.id = 0
        self.x_coord = 0
        self.y_coord = 0
        self.demand = 0
    def __repr__(self):
        return f"Node(id={self.id}, x_coord={self.x_coord}, y_coord={self.y_coord}, demand={self.demand})"

# --- Model Class Definition ---
class Model():
    def __init__(self):
        self.best_sol = None
        self.node_id_list = []
        self.demand_dict = {}
        self.depot = None
        self.number_of_demands = 0
        self.vehicle_cap = 160
        self.distance_matrix = None  
        self.distance_matrix_dict = {}  
        # --- ALNS Parameters (will be set later) ---
        self.rand_d_max = 0.4
        self.rand_d_min = 0.1
        self.worst_d_min = 5
        self.worst_d_max = 20
        self.regret_n = 2
        self.r1 = 30
        self.r2 = 18
        self.r3 = 12
        self.r4 = 0
        self.rho = 0.6
        self.phi = 0.9
        self.max_non_imp = 1000
        self.epochs = 1501 # Default epochs
        # --- Operator Weights/Scores (can be part of Model or managed per run) ---
        self.max_destroy_id = 2
        self.max_repair_id = 3
        self.d_weight = np.ones(2) * 10
        self.d_score = np.zeros(2)
        self.r_weight = np.ones(3) * 10
        self.r_score = np.zeros(3)
        self.dr_weight = np.ones(6) * 10
        self.dr_score = np.zeros(6)
        self.rt_weight = np.ones((2, 3)) * 10
        self.rt_score = np.zeros((2, 3))

# --- Data Reading Functions (readCsvFile, readDateFile) ---
def readDateFile(index, model):
    # Index 0: CMT1.vrp.txt
    # Index 1: CMT2.vrp.txt
    # Index 2: CMT12.vrp.txt
    # Index 3: CMT5.vrp.txt
    # Index 4: CMT3.vrp.txt
    # Index 5: CMT4.vrp.txt
    # Index 6: CMT11.vrp.txt
    depot = [
        [30.0, 40.0], [40.0, 40.0], [40.0, 50.0], [35.0, 35.0], [35.0, 35.0],
        [35.0, 35.0], [10.0, 45.0]
    ]
    customers_set = [
            [
                [37.0, 49.0, 52.0, 20.0, 40.0, 21.0, 17.0, 31.0, 52.0, 51.0,
                 42.0, 31.0, 5.0, 12.0, 36.0, 52.0, 27.0, 17.0, 13.0, 57.0,
                 62.0, 42.0, 16.0, 8.0, 7.0, 27.0, 30.0, 43.0, 58.0, 58.0,
                 37.0, 38.0, 46.0, 61.0, 62.0, 63.0, 32.0, 45.0, 59.0, 5.0,
                 10.0, 21.0, 5.0, 30.0, 39.0, 32.0, 25.0, 25.0, 48.0, 56.0],
                [52.0, 49.0, 64.0, 26.0, 30.0, 47.0, 63.0, 62.0, 33.0, 21.0,
                 41.0, 32.0, 25.0, 42.0, 16.0, 41.0, 23.0, 33.0, 13.0, 58.0,
                 42.0, 57.0, 57.0, 52.0, 38.0, 68.0, 48.0, 67.0, 48.0, 27.0,
                 69.0, 46.0, 10.0, 33.0, 63.0, 69.0, 22.0, 35.0, 15.0, 6.0,
                 17.0, 10.0, 64.0, 15.0, 10.0, 39.0, 32.0, 55.0, 28.0, 37.0]
            ],
            [
                [22.0, 36.0, 21.0, 45.0, 55.0, 33.0, 50.0, 55.0, 26.0, 40.0,
                 55.0, 35.0, 62.0, 62.0, 62.0, 21.0, 33.0, 9.0, 62.0, 66.0,
                 44.0, 26.0, 11.0, 7.0, 17.0, 41.0, 55.0, 35.0, 52.0, 43.0,
                 31.0, 22.0, 26.0, 50.0, 55.0, 54.0, 60.0, 47.0, 30.0, 30.0,
                 12.0, 15.0, 16.0, 21.0, 50.0, 51.0, 50.0, 48.0, 12.0, 15.0,
                 29.0, 54.0, 55.0, 67.0, 10.0, 6.0, 65.0, 40.0, 70.0, 64.0,
                 36.0, 30.0, 20.0, 15.0, 50.0, 57.0, 45.0, 38.0, 50.0, 66.0,
                 59.0, 35.0, 27.0, 40.0, 40.0],
                [22.0, 26.0, 45.0, 35.0, 20.0, 34.0, 50.0, 45.0, 59.0, 66.0,
                 65.0, 51.0, 35.0, 57.0, 24.0, 36.0, 44.0, 56.0, 48.0, 14.0,
                 13.0, 13.0, 28.0, 43.0, 64.0, 46.0, 34.0, 16.0, 26.0, 26.0,
                 76.0, 53.0, 29.0, 40.0, 50.0, 10.0, 15.0, 66.0, 60.0, 50.0,
                 17.0, 14.0, 19.0, 48.0, 30.0, 42.0, 15.0, 21.0, 38.0, 56.0,
                 39.0, 38.0, 57.0, 41.0, 70.0, 25.0, 27.0, 60.0, 64.0, 4.0,
                 6.0, 20.0, 30.0, 5.0, 70.0, 72.0, 42.0, 33.0, 4.0, 8.0,
                 5.0, 60.0, 24.0, 20.0, 37.0]
            ],
            [
                [45.0, 45.0, 42.0, 42.0, 42.0, 40.0, 40.0, 38.0, 38.0, 35.0,
                 35.0, 25.0, 22.0, 22.0, 20.0, 20.0, 18.0, 15.0, 15.0, 30.0,
                 30.0, 28.0, 28.0, 25.0, 25.0, 25.0, 23.0, 23.0, 20.0, 20.0,
                 10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 2.0, 0.0, 0.0, 35.0,
                 35.0, 33.0, 33.0, 32.0, 30.0, 30.0, 30.0, 28.0, 28.0, 26.0,
                 25.0, 25.0, 44.0, 42.0, 42.0, 40.0, 40.0, 38.0, 38.0, 35.0,
                 50.0, 50.0, 50.0, 48.0, 48.0, 47.0, 47.0, 45.0, 45.0, 95.0,
                 95.0, 53.0, 92.0, 53.0, 45.0, 90.0, 88.0, 88.0, 87.0, 85.0,
                 85.0, 75.0, 72.0, 70.0, 68.0, 66.0, 65.0, 65.0, 63.0, 60.0,
                 60.0, 67.0, 65.0, 65.0, 62.0, 60.0, 60.0, 58.0, 55.0, 55.0],
                [68.0, 70.0, 66.0, 68.0, 65.0, 69.0, 66.0, 68.0, 70.0, 66.0,
                 69.0, 85.0, 75.0, 85.0, 80.0, 85.0, 75.0, 75.0, 80.0, 50.0,
                 52.0, 52.0, 55.0, 50.0, 52.0, 55.0, 52.0, 55.0, 50.0, 55.0,
                 35.0, 40.0, 40.0, 45.0, 35.0, 45.0, 40.0, 40.0, 45.0, 30.0,
                 32.0, 32.0, 35.0, 30.0, 30.0, 32.0, 35.0, 30.0, 35.0, 32.0,
                 30.0, 35.0, 5.0, 10.0, 15.0, 5.0, 15.0, 5.0, 15.0, 5.0,
                 30.0, 35.0, 40.0, 30.0, 40.0, 35.0, 40.0, 30.0, 35.0, 30.0,
                 35.0, 30.0, 30.0, 35.0, 65.0, 35.0, 30.0, 35.0, 30.0, 25.0,
                 35.0, 55.0, 55.0, 58.0, 60.0, 55.0, 55.0, 60.0, 58.0, 55.0,
                 60.0, 85.0, 85.0, 82.0, 80.0, 80.0, 85.0, 75.0, 80.0, 85.0]
            ],
            [
                [22.0, 36.0, 21.0, 45.0, 55.0, 33.0, 50.0, 55.0, 26.0, 40.0,
                 55.0, 35.0, 62.0, 62.0, 62.0, 21.0, 33.0, 9.0, 62.0, 66.0,
                 44.0, 26.0, 11.0, 7.0, 17.0, 41.0, 55.0, 35.0, 52.0, 43.0,
                 31.0, 22.0, 26.0, 50.0, 55.0, 54.0, 60.0, 47.0, 30.0, 30.0,
                 12.0, 15.0, 16.0, 21.0, 50.0, 51.0, 50.0, 48.0, 12.0, 37.0,
                 49.0, 52.0, 20.0, 40.0, 21.0, 17.0, 31.0, 52.0, 51.0, 42.0,
                 31.0, 5.0, 12.0, 36.0, 52.0, 27.0, 17.0, 13.0, 57.0, 62.0,
                 42.0, 16.0, 8.0, 7.0, 27.0, 30.0, 43.0, 58.0, 58.0, 37.0,
                 38.0, 46.0, 61.0, 62.0, 63.0, 32.0, 45.0, 59.0, 5.0, 10.0,
                 21.0, 5.0, 30.0, 39.0, 32.0, 25.0, 25.0, 48.0, 56.0, 41.0,
                 35.0, 55.0, 55.0, 15.0, 25.0, 20.0, 10.0, 55.0, 30.0, 20.0,
                 50.0, 30.0, 15.0, 30.0, 10.0, 5.0, 20.0, 15.0, 45.0, 45.0,
                 45.0, 55.0, 65.0, 65.0, 45.0, 35.0, 41.0, 64.0, 40.0, 31.0,
                 35.0, 53.0, 65.0, 63.0, 2.0, 20.0, 5.0, 60.0, 40.0, 42.0,
                 24.0, 23.0, 11.0, 6.0, 2.0, 8.0, 13.0, 6.0, 47.0, 49.0,
                 27.0, 37.0, 57.0, 63.0, 53.0, 32.0, 36.0, 21.0, 17.0, 12.0,
                 24.0, 27.0, 15.0, 62.0, 49.0, 67.0, 56.0, 37.0, 37.0, 57.0,
                 47.0, 44.0, 46.0, 49.0, 49.0, 53.0, 61.0, 57.0, 56.0, 55.0,
                 15.0, 14.0, 11.0, 16.0, 4.0, 28.0, 26.0, 26.0, 31.0, 15.0,
                 22.0, 18.0, 26.0, 25.0, 22.0, 25.0, 19.0, 20.0, 18.0],
                [22.0, 26.0, 45.0, 35.0, 20.0, 34.0, 50.0, 45.0, 59.0, 66.0,
                 65.0, 51.0, 35.0, 57.0, 24.0, 36.0, 44.0, 56.0, 48.0, 14.0,
                 13.0, 13.0, 28.0, 43.0, 64.0, 46.0, 34.0, 16.0, 26.0, 26.0,
                 76.0, 53.0, 29.0, 40.0, 50.0, 10.0, 15.0, 66.0, 60.0, 50.0,
                 17.0, 14.0, 19.0, 48.0, 30.0, 42.0, 15.0, 21.0, 38.0, 52.0,
                 49.0, 64.0, 26.0, 30.0, 47.0, 63.0, 62.0, 33.0, 21.0, 41.0,
                 32.0, 25.0, 42.0, 16.0, 41.0, 23.0, 33.0, 13.0, 58.0, 42.0,
                 57.0, 57.0, 52.0, 38.0, 68.0, 48.0, 67.0, 48.0, 27.0, 69.0,
                 46.0, 10.0, 33.0, 63.0, 69.0, 22.0, 35.0, 15.0, 6.0, 17.0,
                 10.0, 64.0, 15.0, 10.0, 39.0, 32.0, 55.0, 28.0, 37.0, 49.0,
                 17.0, 45.0, 20.0, 30.0, 30.0, 50.0, 43.0, 60.0, 60.0, 65.0,
                 35.0, 25.0, 10.0, 5.0, 20.0, 30.0, 40.0, 60.0, 65.0, 20.0,
                 10.0, 5.0, 35.0, 20.0, 30.0, 40.0, 37.0, 42.0, 60.0, 52.0,
                 69.0, 52.0, 55.0, 65.0, 60.0, 20.0, 5.0, 12.0, 25.0, 7.0,
                 12.0, 3.0, 14.0, 38.0, 48.0, 56.0, 52.0, 68.0, 47.0, 58.0,
                 43.0, 31.0, 29.0, 23.0, 12.0, 12.0, 26.0, 24.0, 34.0, 24.0,
                 58.0, 69.0, 77.0, 77.0, 73.0, 5.0, 39.0, 47.0, 56.0, 68.0,
                 16.0, 17.0, 13.0, 11.0, 42.0, 43.0, 52.0, 48.0, 37.0, 54.0,
                 47.0, 37.0, 31.0, 22.0, 18.0, 18.0, 52.0, 35.0, 67.0, 19.0,
                 22.0, 24.0, 27.0, 24.0, 27.0, 21.0, 21.0, 26.0, 18.0]
            ],
            [
                [41.0, 35.0, 55.0, 55.0, 15.0, 25.0, 20.0, 10.0, 55.0, 30.0,
                 20.0, 50.0, 30.0, 15.0, 30.0, 10.0, 5.0, 20.0, 15.0, 45.0,
                 45.0, 45.0, 55.0, 65.0, 65.0, 45.0, 35.0, 41.0, 64.0, 40.0,
                 31.0, 35.0, 53.0, 65.0, 63.0, 2.0, 20.0, 5.0, 60.0, 40.0,
                 42.0, 24.0, 23.0, 11.0, 6.0, 2.0, 8.0, 13.0, 6.0, 47.0,
                 49.0, 27.0, 37.0, 57.0, 63.0, 53.0, 32.0, 36.0, 21.0, 17.0,
                 12.0, 24.0, 27.0, 15.0, 62.0, 49.0, 67.0, 56.0, 37.0, 37.0,
                 57.0, 47.0, 44.0, 46.0, 49.0, 49.0, 53.0, 61.0, 57.0, 56.0,
                 55.0, 15.0, 14.0, 11.0, 16.0, 4.0, 28.0, 26.0, 26.0, 31.0,
                 15.0, 22.0, 18.0, 26.0, 25.0, 22.0, 25.0, 19.0, 20.0, 18.0],
                [49.0, 17.0, 45.0, 20.0, 30.0, 30.0, 50.0, 43.0, 60.0, 60.0,
                 65.0, 35.0, 25.0, 10.0, 5.0, 20.0, 30.0, 40.0, 60.0, 65.0,
                 20.0, 10.0, 5.0, 35.0, 20.0, 30.0, 40.0, 37.0, 42.0, 60.0,
                 52.0, 69.0, 52.0, 55.0, 65.0, 60.0, 20.0, 5.0, 12.0, 25.0,
                 7.0, 12.0, 3.0, 14.0, 38.0, 48.0, 56.0, 52.0, 68.0, 47.0,
                 58.0, 43.0, 31.0, 29.0, 23.0, 12.0, 12.0, 26.0, 24.0, 34.0,
                 24.0, 58.0, 69.0, 77.0, 77.0, 73.0, 5.0, 39.0, 47.0, 56.0,
                 68.0, 16.0, 17.0, 13.0, 11.0, 42.0, 43.0, 52.0, 48.0, 37.0,
                 54.0, 47.0, 37.0, 31.0, 22.0, 18.0, 18.0, 52.0, 35.0, 67.0,
                 19.0, 22.0, 24.0, 27.0, 24.0, 27.0, 21.0, 21.0, 26.0, 18.0]
            ],
            [
                [37.0, 49.0, 52.0, 20.0, 40.0, 21.0, 17.0, 31.0, 52.0, 51.0,
                 42.0, 31.0, 5.0, 12.0, 36.0, 52.0, 27.0, 17.0, 13.0, 57.0,
                 62.0, 42.0, 16.0, 8.0, 7.0, 27.0, 30.0, 43.0, 58.0, 58.0,
                 37.0, 38.0, 46.0, 61.0, 62.0, 63.0, 32.0, 45.0, 59.0, 5.0,
                 10.0, 21.0, 5.0, 30.0, 39.0, 32.0, 25.0, 25.0, 48.0, 56.0,
                 41.0, 35.0, 55.0, 55.0, 15.0, 25.0, 20.0, 10.0, 55.0, 30.0,
                 20.0, 50.0, 30.0, 15.0, 30.0, 10.0, 5.0, 20.0, 15.0, 45.0,
                 45.0, 45.0, 55.0, 65.0, 65.0, 45.0, 35.0, 41.0, 64.0, 40.0,
                 31.0, 35.0, 53.0, 65.0, 63.0, 2.0, 20.0, 5.0, 60.0, 40.0,
                 42.0, 24.0, 23.0, 11.0, 6.0, 2.0, 8.0, 13.0, 6.0, 47.0,
                 49.0, 27.0, 37.0, 57.0, 63.0, 53.0, 32.0, 36.0, 21.0, 17.0,
                 12.0, 24.0, 27.0, 15.0, 62.0, 49.0, 67.0, 56.0, 37.0, 37.0,
                 57.0, 47.0, 44.0, 46.0, 49.0, 49.0, 53.0, 61.0, 57.0, 56.0,
                 55.0, 15.0, 14.0, 11.0, 16.0, 4.0, 28.0, 26.0, 26.0, 31.0,
                 15.0, 22.0, 18.0, 26.0, 25.0, 22.0, 25.0, 19.0, 20.0, 18.0],
                [52.0, 49.0, 64.0, 26.0, 30.0, 47.0, 63.0, 62.0, 33.0, 21.0,
                 41.0, 32.0, 25.0, 42.0, 16.0, 41.0, 23.0, 33.0, 13.0, 58.0,
                 42.0, 57.0, 57.0, 52.0, 38.0, 68.0, 48.0, 67.0, 48.0, 27.0,
                 69.0, 46.0, 10.0, 33.0, 63.0, 69.0, 22.0, 35.0, 15.0, 6.0,
                 17.0, 10.0, 64.0, 15.0, 10.0, 39.0, 32.0, 55.0, 28.0, 37.0,
                 49.0, 17.0, 45.0, 20.0, 30.0, 30.0, 50.0, 43.0, 60.0, 60.0,
                 65.0, 35.0, 25.0, 10.0, 5.0, 20.0, 30.0, 40.0, 60.0, 65.0,
                 20.0, 10.0, 5.0, 35.0, 20.0, 30.0, 40.0, 37.0, 42.0, 60.0,
                 52.0, 69.0, 52.0, 55.0, 65.0, 60.0, 20.0, 5.0, 12.0, 25.0,
                 7.0, 12.0, 3.0, 14.0, 38.0, 48.0, 56.0, 52.0, 68.0, 47.0,
                 58.0, 43.0, 31.0, 29.0, 23.0, 12.0, 12.0, 26.0, 24.0, 34.0,
                 24.0, 58.0, 69.0, 77.0, 77.0, 73.0, 5.0, 39.0, 47.0, 56.0,
                 68.0, 16.0, 17.0, 13.0, 11.0, 42.0, 43.0, 52.0, 48.0, 37.0,
                 54.0, 47.0, 37.0, 31.0, 22.0, 18.0, 18.0, 52.0, 35.0, 67.0,
                 19.0, 22.0, 24.0, 27.0, 24.0, 27.0, 21.0, 21.0, 26.0, 18.0]
            ],
            [
                [25.0, 25.0, 31.0, 32.0, 31.0, 32.0, 34.0, 46.0, 35.0, 34.0,
                 35.0, 47.0, 40.0, 39.0, 36.0, 73.0, 73.0, 24.0, 76.0, 76.0,
                 76.0, 78.0, 78.0, 79.0, 79.0, 79.0, 82.0, 82.0, 90.0, 84.0,
                 84.0, 84.0, 85.0, 87.0, 85.0, 87.0, 86.0, 86.0, 86.0, 85.0,
                 89.0, 89.0, 89.0, 92.0, 92.0, 94.0, 94.0, 94.0, 96.0, 99.0,
                 99.0, 83.0, 83.0, 85.0, 85.0, 85.0, 87.0, 87.0, 90.0, 90.0,
                 93.0, 93.0, 93.0, 94.0, 95.0, 99.0, 37.0, 50.0, 35.0, 35.0,
                 44.0, 46.0, 46.0, 46.0, 46.0, 48.0, 50.0, 50.0, 54.0, 54.0,
                 10.0, 10.0, 18.0, 17.0, 16.0, 14.0, 15.0, 11.0, 18.0, 21.0,
                 20.0, 18.0, 20.0, 22.0, 16.0, 20.0, 25.0, 30.0, 20.0, 22.0,
                 18.0, 16.0, 20.0, 18.0, 14.0, 15.0, 16.0, 28.0, 33.0, 30.0,
                 13.0, 15.0, 18.0, 25.0, 30.0, 25.0, 16.0, 25.0, 5.0, 5.0],
                [1.0, 3.0, 5.0, 5.0, 7.0, 9.0, 9.0, 9.0, 7.0, 6.0,
                 5.0, 6.0, 5.0, 3.0, 3.0, 6.0, 8.0, 36.0, 6.0, 10.0,
                 13.0, 3.0, 9.0, 3.0, 5.0, 11.0, 3.0, 7.0, 15.0, 3.0,
                 5.0, 9.0, 1.0, 5.0, 8.0, 7.0, 41.0, 44.0, 46.0, 55.0,
                 43.0, 46.0, 52.0, 42.0, 52.0, 42.0, 44.0, 48.0, 42.0, 46.0,
                 50.0, 80.0, 83.0, 81.0, 85.0, 89.0, 80.0, 86.0, 77.0, 88.0,
                 82.0, 84.0, 89.0, 86.0, 80.0, 89.0, 83.0, 80.0, 85.0, 87.0,
                 86.0, 89.0, 83.0, 87.0, 89.0, 83.0, 85.0, 88.0, 86.0, 90.0,
                 35.0, 40.0, 30.0, 35.0, 38.0, 40.0, 42.0, 42.0, 40.0, 39.0,
                 40.0, 41.0, 44.0, 44.0, 45.0, 45.0, 45.0, 55.0, 50.0, 51.0,
                 49.0, 48.0, 55.0, 53.0, 50.0, 51.0, 54.0, 33.0, 38.0, 50.0,
                 40.0, 36.0, 31.0, 37.0, 46.0, 52.0, 33.0, 35.0, 40.0, 50.0]
            ]
    ]

    demands_set = [
            [7, 30, 16, 9, 21, 15, 19, 23, 11, 5,
             19, 29, 23, 21, 10, 15, 3, 41, 9, 28,
             8, 8, 16, 10, 28, 7, 15, 14, 6, 19,
             11, 12, 23, 26, 17, 6, 9, 15, 14, 7,
             27, 13, 11, 16, 10, 5, 25, 17, 18, 10],
            [18, 26, 11, 30, 21, 19, 15, 16, 29, 26,
             37, 16, 12, 31, 8, 19, 20, 13, 15, 22,
             28, 12, 6, 27, 14, 18, 17, 29, 13, 22,
             25, 28, 27, 19, 10, 12, 14, 24, 16, 33,
             15, 11, 18, 17, 21, 27, 19, 20, 5, 22,
             12, 19, 22, 16, 7, 26, 14, 21, 24, 13,
             15, 18, 11, 28, 9, 37, 30, 10, 8, 11,
             3, 1, 6, 10, 20],
            [10, 30, 10, 10, 10, 20, 20, 20, 10, 10,
             10, 20, 30, 10, 40, 40, 20, 20, 10, 10,
             20, 20, 10, 10, 40, 10, 10, 20, 10, 10,
             20, 30, 40, 20, 10, 10, 20, 30, 20, 10,
             10, 20, 10, 10, 10, 30, 10, 10, 10, 10,
             10, 10, 20, 40, 10, 30, 40, 30, 10, 20,
             10, 20, 50, 10, 10, 10, 10, 10, 10, 30,
             20, 10, 10, 50, 20, 10, 10, 20, 10, 10,
             30, 20, 10, 20, 30, 10, 20, 30, 10, 10,
             10, 20, 40, 10, 30, 10, 30, 20, 10, 20],
            [18, 26, 11, 30, 21, 19, 15, 16, 29, 26,
             37, 16, 12, 31, 8, 19, 20, 13, 15, 22,
             28, 12, 6, 27, 14, 18, 17, 29, 13, 22,
             25, 28, 27, 19, 10, 12, 14, 24, 16, 33,
             15, 11, 18, 17, 21, 27, 19, 20, 5, 7,
             30, 16, 9, 21, 15, 19, 23, 11, 5, 19,
             29, 23, 21, 10, 15, 3, 41, 9, 28, 8,
             8, 16, 10, 28, 7, 15, 14, 6, 19, 11,
             12, 23, 26, 17, 6, 9, 15, 14, 7, 27,
             13, 11, 16, 10, 5, 25, 17, 18, 10, 10,
             7, 13, 19, 26, 3, 5, 9, 16, 16, 12,
             19, 23, 20, 8, 19, 2, 12, 17, 9, 11,
             18, 29, 3, 6, 17, 16, 16, 9, 21, 27,
             23, 11, 14, 8, 5, 8, 16, 31, 9, 5,
             5, 7, 18, 16, 1, 27, 36, 30, 13, 10,
             9, 14, 18, 2, 6, 7, 18, 28, 3, 13,
             19, 10, 9, 20, 25, 25, 36, 6, 5, 15,
             25, 9, 8, 18, 13, 14, 3, 23, 6, 26,
             16, 11, 7, 41, 35, 26, 9, 15, 3, 1,
             2, 22, 27, 20, 11, 12, 10, 9, 17],
            [10, 7, 13, 19, 26, 3, 5, 9, 16, 16,
             12, 19, 23, 20, 8, 19, 2, 12, 17, 9,
             11, 18, 29, 3, 6, 17, 16, 16, 9, 21,
             27, 23, 11, 14, 8, 5, 8, 16, 31, 9,
             5, 5, 7, 18, 16, 1, 27, 36, 30, 13,
             10, 9, 14, 18, 2, 6, 7, 18, 28, 3,
             13, 19, 10, 9, 20, 25, 25, 36, 6, 5,
             15, 25, 9, 8, 18, 13, 14, 3, 23, 6,
             26, 16, 11, 7, 41, 35, 26, 9, 15, 3,
             1, 2, 22, 27, 20, 11, 12, 10, 9, 17],
            [7, 30, 16, 9, 21, 15, 19, 23, 11, 5,
             19, 29, 23, 21, 10, 15, 3, 41, 9, 28,
             8, 8, 16, 10, 28, 7, 15, 14, 6, 19,
             11, 12, 23, 26, 17, 6, 9, 15, 14, 7,
             27, 13, 11, 16, 10, 5, 25, 17, 18, 10,
             10, 7, 13, 19, 26, 3, 5, 9, 16, 16,
             12, 19, 23, 20, 8, 19, 2, 12, 17, 9,
             11, 18, 29, 3, 6, 17, 16, 16, 9, 21,
             27, 23, 11, 14, 8, 5, 8, 16, 31, 9,
             5, 5, 7, 18, 16, 1, 27, 36, 30, 13,
             10, 9, 14, 18, 2, 6, 7, 18, 28, 3,
             13, 19, 10, 9, 20, 25, 25, 36, 6, 5,
             15, 25, 9, 8, 18, 13, 14, 3, 23, 6,
             26, 16, 11, 7, 41, 35, 26, 9, 15, 3,
             1, 2, 22, 27, 20, 11, 12, 10, 9, 17],
            [25, 7, 13, 6, 14, 5, 11, 19, 5, 15,
             15, 17, 13, 12, 18, 13, 18, 12, 17, 4,
             7, 12, 13, 8, 16, 15, 6, 5, 9, 11,
             10, 3, 7, 2, 4, 4, 18, 14, 12, 17,
             20, 14, 16, 10, 9, 11, 7, 13, 5, 4,
             21, 13, 11, 12, 14, 10, 8, 16, 19, 5,
             17, 7, 16, 14, 17, 13, 17, 13, 14, 16,
             7, 13, 9, 11, 35, 5, 28, 7, 3, 10,
             7, 12, 11, 10, 8, 11, 21, 4, 15, 16,
             4, 16, 7, 10, 9, 11, 17, 12, 11, 7,
             9, 11, 12, 7, 8, 6, 5, 12, 13, 7,
             7, 8, 11, 13, 11, 10, 7, 4, 20, 13]]

    capacities = [
        160, 140, 200, 200, 200, 200, 200
    ]

    model.vehicle_cap = capacities[index]
    node = Node()
    node.id = 0
    node.x_coord = depot[index][0]
    node.y_coord = depot[index][1]
    node.demand = 0
    model.depot = node
    model.demand_dict[node.id] = node
    length = len(demands_set[index])
    for i in range(length):
        node = Node()
        node.x_coord = customers_set[index][0][i]
        node.y_coord = customers_set[index][1][i]
        node.demand = demands_set[index][i]
        node.id = i + 1
        model.node_id_list.append(node.id)
        model.demand_dict[node.id] = node
    model.number_of_demands = len(model.demand_dict) - 1


# --- Core ALNS Functions (calDistanceMatrix, genInitialSol, splitRoutes, etc.) ---
def calDistanceMatrix(model):
    ids = [model.depot.id] + model.node_id_list              # [0, 1, 2, …]
    coords = np.array([[model.demand_dict[i].x_coord,
                        model.demand_dict[i].y_coord] for i in ids],
                      dtype=np.float32)                      # shape=(n+1,2)

    diff = coords[:, None, :] - coords[None, :, :]           # (n+1,n+1,2)
    dist_mat = np.sqrt((diff ** 2).sum(-1, dtype=np.float32))  # (n+1,n+1)

    model.distance_matrix = dist_mat                         # ndarray
    model.id2idx = {id_: idx for idx, id_ in enumerate(ids)}
    model.idx2id = ids                                        # list

    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):
            model.distance_matrix_dict[(id_i, id_j)] = dist_mat[i, j] \
                if hasattr(model, "distance_matrix_dict") else None

def genInitialSol(node_no_seq):
    node_no_seq = copy.deepcopy(node_no_seq)
    random.shuffle(node_no_seq)
    return node_no_seq


try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False


def _split_dp_core_py(seq, dem_arr, Q, dist_mat, depot_idx):

    n = seq.shape[0]
    d0 = np.empty(n, dtype=np.float32)
    dr = np.empty(n, dtype=np.float32)
    for t in range(n):
        idx = seq[t]
        d0[t] = dist_mat[depot_idx, idx]
        dr[t] = dist_mat[idx, depot_idx]

    F    = np.full(n + 1, np.inf, dtype=np.float32)
    pred = np.zeros(n + 1, dtype=np.int32)
    F[0] = 0.0

    for i in range(1, n + 1):
        load = 0
        cost = 0.0
        for j in range(i, 0, -1):
            load += dem_arr[j - 1]
            if load > Q:
                break
            if j == i:
                cost = d0[j - 1] + dr[j - 1]
            else:
                prev_idx = seq[j]
                cur_idx  = seq[j - 1]
                cost = cost \
                       - dist_mat[prev_idx, depot_idx] \
                       + dist_mat[prev_idx, cur_idx]  \
                       + dist_mat[cur_idx,  depot_idx]

            new_cost = F[j - 1] + cost
            if new_cost < F[i]:
                F[i]   = new_cost
                pred[i]= j - 1
    return pred, float(F[n])


if _NUMBA:
    _split_dp_core = njit(_split_dp_core_py, cache=True, fastmath=True)
else:
    _split_dp_core = _split_dp_core_py


def splitRoutes(node_no_seq, model):

    if not node_no_seq:
        return [], 0.0

    seq_idx = np.array([model.id2idx[id_] for id_ in node_no_seq], dtype=np.int32)
    dem_arr = np.array([model.demand_dict[id_].demand for id_ in node_no_seq],
                       dtype=np.int32)

    pred, total_cost = _split_dp_core(
        seq_idx,
        dem_arr,
        model.vehicle_cap,
        model.distance_matrix,
        depot_idx=0
    )

    routes_idx = []
    i = len(seq_idx)
    while i > 0:
        j = int(pred[i]) + 1
        routes_idx.append(seq_idx[j - 1:i])
        i = int(pred[i])

    routes_idx.reverse()

    depot_id = model.depot.id
    route_list = [
        [depot_id] + [model.idx2id[int(idx)] for idx in r] + [depot_id]
        for r in routes_idx
    ]
    return route_list, total_cost


def calRouteDistance(route, model):
    return sum(model.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))


def calObj(node_no_seq, model):
    route_list, total_distance = splitRoutes(node_no_seq, model)
    if not route_list:
        return float('inf'), [], []
    route_distance = []
    for route in route_list:
        idx = [model.id2idx[id_] for id_ in route]
        d = model.distance_matrix[idx[:-1], idx[1:]].sum()
        route_distance.append(float(d))

    return float(total_distance), route_list, route_distance

def createRandomDestory(model):
    if not model.node_id_list: return []
    num_nodes = len(model.node_id_list)
    if num_nodes <= 1: return []
    d = random.uniform(model.rand_d_min, model.rand_d_max)
    num_to_remove = min(max(1, int(d * num_nodes)), num_nodes - 1 if num_nodes > 1 else 0)
    if num_to_remove == 0 and num_nodes > 0: num_to_remove = 1
    if num_to_remove == 0 : return []
    k = min(num_to_remove, len(model.node_id_list))
    return random.sample(model.node_id_list, k)


def createWorseDestory(model, sol):
    deta_f = []
    if not sol.node_no_seq: return []
    current_obj = sol.obj
    if current_obj == float('inf'): return []
    nodes_to_consider = copy.deepcopy(sol.node_no_seq)
    for node_no in nodes_to_consider:
        node_no_seq_ = [n for n in nodes_to_consider if n != node_no]
        if not node_no_seq_:
             obj = 0
        else:
             obj, _, _ = calObj(node_no_seq_, model)
        if obj == float('inf'):
            delta = -float('inf')
        else:
            delta = obj - current_obj
            if obj == float('inf'):
                 delta = -float('inf')
            else:
                 delta = current_obj - obj
        deta_f.append(delta)
    node_delta_pairs = list(zip(nodes_to_consider, deta_f))
    node_delta_pairs.sort(key=lambda item: item[1], reverse=True)

    # Determine number to remove
    num_nodes = len(nodes_to_consider)
    d_min = min(model.worst_d_min, num_nodes - 1 if num_nodes > 1 else 0)
    d_max = min(model.worst_d_max, num_nodes - 1 if num_nodes > 1 else 0)
    if d_min < 0: d_min = 0
    if d_max < d_min: d_max = d_min
    if num_nodes > 1 and d_max == 0: d_max = 1 # Ensure at least 1 if possible
    if d_min > d_max : d_min = d_max

    num_to_remove = 0
    if d_max > 0:
        num_to_remove = random.randint(d_min, d_max)
        num_to_remove = min(num_to_remove, len(node_delta_pairs), num_nodes - 1 if num_nodes > 1 else 0)
        if num_to_remove < 0: num_to_remove = 0

    if num_to_remove == 0: return []

    # Select the top 'num_to_remove' nodes based on sorted delta_f
    return [pair[0] for pair in node_delta_pairs[:num_to_remove]]



def createRandomRepair(remove_list, model, sol):
    unassigned_node_no_seq = copy.deepcopy(remove_list)
    assigned_node_no_seq = [node_no for node_no in sol.node_no_seq if node_no not in unassigned_node_no_seq]

    for node_no in unassigned_node_no_seq:
        insert_limit = len(assigned_node_no_seq)
        index = random.randint(0, insert_limit)
        assigned_node_no_seq.insert(index, node_no)

    new_sol = Sol()
    new_sol.node_no_seq = assigned_node_no_seq
    new_sol.obj, new_sol.route_list, new_sol.route_distance = calObj(assigned_node_no_seq, model)
    return new_sol


def findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model):
    best_insert_node_no = None
    best_insert_index = None
    best_insert_cost_increase = float('inf')
    if not assigned_node_no_seq:
         base_obj = 0
    else:
        base_obj_result = calObj(assigned_node_no_seq, model)
        base_obj = base_obj_result[0]
        if base_obj == float('inf'):
            pass

    for node_no in unassigned_node_no_seq:
        node_best_index = -1
        node_best_cost_increase = float('inf')
        for i in range(len(assigned_node_no_seq) + 1):
            assigned_node_no_seq_ = assigned_node_no_seq[:i] + [node_no] + assigned_node_no_seq[i:]
            obj_, _, _ = calObj(assigned_node_no_seq_, model)
            if obj_ == float('inf'):
                cost_increase = float('inf')
            elif base_obj == float('inf'):
                cost_increase = -float('inf')
            else:
                cost_increase = obj_ - base_obj
            if cost_increase < node_best_cost_increase:
                node_best_cost_increase = cost_increase
                node_best_index = i
        if node_best_cost_increase < best_insert_cost_increase:
            best_insert_cost_increase = node_best_cost_increase
            best_insert_index = node_best_index
            best_insert_node_no = node_no

    if best_insert_node_no is None and unassigned_node_no_seq:
        return None, None

    return best_insert_node_no, best_insert_index



def createGreedyRepair(remove_list, model, sol):
    unassigned_node_no_seq = copy.deepcopy(remove_list)
    assigned_node_no_seq = [node_no for node_no in sol.node_no_seq if node_no not in unassigned_node_no_seq]

    while len(unassigned_node_no_seq) > 0:
        insert_node_no, insert_index = findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
        if insert_node_no is None:
            break
        assigned_node_no_seq.insert(insert_index, insert_node_no)
        unassigned_node_no_seq.remove(insert_node_no)

    new_sol = Sol()
    new_sol.node_no_seq = assigned_node_no_seq
    new_sol.obj, new_sol.route_list, new_sol.route_distance = calObj(assigned_node_no_seq, model)
    return new_sol


def findRegretInsert(unassigned_node_no_seq, assigned_node_no_seq, model):
    opt_insert_node_no = None
    opt_insert_index = None
    opt_node_max_regret = -float('inf')

    if not assigned_node_no_seq:
         base_obj = 0
    else:
        base_obj_result = calObj(assigned_node_no_seq, model)
        base_obj = base_obj_result[0]
        if base_obj == float('inf'):
             pass
    candidate_insertions = {}

    for node_no in unassigned_node_no_seq:
        insertion_options = []
        for i in range(len(assigned_node_no_seq) + 1):
            assigned_node_no_seq_ = assigned_node_no_seq[:i] + [node_no] + assigned_node_no_seq[i:]
            obj_, _, _ = calObj(assigned_node_no_seq_, model)

            if obj_ == float('inf'):
                cost_increase = float('inf')
            elif base_obj == float('inf'):
                 cost_increase = -float('inf') if obj_ != float('inf') else float('inf')
            else:
                cost_increase = obj_ - base_obj

            if cost_increase != float('inf'):
                 insertion_options.append((cost_increase, i))
        insertion_options.sort(key=lambda x: x[0])

        if insertion_options:
             candidate_insertions[node_no] = insertion_options

    node_regrets = {}
    for node_no, options in candidate_insertions.items():
        regret_val = 0
        best_cost_increase = options[0][0]

        for k in range(1, min(model.regret_n, len(options))):
            regret_val += (options[k][0] - best_cost_increase)

        node_regrets[node_no] = regret_val

        if regret_val > opt_node_max_regret:
            opt_node_max_regret = regret_val
            opt_insert_node_no = node_no
            opt_insert_index = options[0][1]

    if opt_insert_node_no is None and unassigned_node_no_seq:
        min_cost_increase = float('inf')
        fallback_node = None
        fallback_index = None
        for node_no, options in candidate_insertions.items():
            if options and options[0][0] < min_cost_increase:
                min_cost_increase = options[0][0]
                fallback_node = node_no
                fallback_index = options[0][1]

        if fallback_node is not None:
             opt_insert_node_no = fallback_node
             opt_insert_index = fallback_index
        else:
             return None, None

    return opt_insert_node_no, opt_insert_index


def createRegretRepair(remove_list, model, sol):
    unassigned_node_no_seq = copy.deepcopy(remove_list)
    assigned_node_no_seq = [node_no for node_no in sol.node_no_seq if node_no not in unassigned_node_no_seq]

    while len(unassigned_node_no_seq) > 0:
        insert_node_no, insert_index = findRegretInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
        if insert_node_no is None:
            insert_node_no, insert_index = findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
            if insert_node_no is None:
                 break

        assigned_node_no_seq.insert(insert_index, insert_node_no)
        unassigned_node_no_seq.remove(insert_node_no)

    new_sol = Sol()
    new_sol.node_no_seq = assigned_node_no_seq
    new_sol.obj, new_sol.route_list, new_sol.route_distance = calObj(assigned_node_no_seq, model)
    return new_sol

def selectDestoryRepair(model, attribute_value, destory_id = None):

    d_weight_sum = sum(model.d_weight)
    if d_weight_sum <= 0:
        model.d_weight = np.ones_like(model.d_weight) * 10
        d_weight_sum = sum(model.d_weight)

    r_weight_sum = sum(model.r_weight)
    if r_weight_sum <= 0:
        model.r_weight = np.ones_like(model.r_weight) * 10
        r_weight_sum = sum(model.r_weight)

    dr_weight_sum = sum(model.dr_weight)
    if dr_weight_sum <= 0:
        model.dr_weight = np.ones_like(model.dr_weight) * 10
        dr_weight_sum = sum(model.dr_weight)

    repair_id = -1 # Initialize repair_id
    destory_and_repair_id = -1 # Initialize

    if attribute_value in ["normal", "reward"]:
        d_cumsumprob = (model.d_weight / d_weight_sum).cumsum()
        d_cumsumprob -= np.random.rand()
        true_indices = np.where(d_cumsumprob > 0)[0]
        if len(true_indices) == 0: destory_id = 0
        else: destory_id = true_indices[0]

        r_cumsumprob = (model.r_weight / r_weight_sum).cumsum()
        r_cumsumprob -= np.random.rand()
        true_indices = np.where(r_cumsumprob > 0)[0]
        if len(true_indices) == 0: repair_id = 0
        else: repair_id = true_indices[0]
        destory_and_repair_id = False

    elif attribute_value == "pair":
        dr_cumsumprob = (model.dr_weight / dr_weight_sum).cumsum()
        dr_cumsumprob -= np.random.rand()
        true_indices = np.where(dr_cumsumprob > 0)[0]
        if len(true_indices) == 0: destory_and_repair_id = 0
        else: destory_and_repair_id = true_indices[0]

        destory_id = destory_and_repair_id // model.max_repair_id
        repair_id = destory_and_repair_id % model.max_repair_id

    elif attribute_value == "table_d":
        print("here")
        d_cumsumprob = (model.d_weight / d_weight_sum).cumsum()
        d_cumsumprob -= np.random.rand()
        true_indices = np.where(d_cumsumprob > 0)[0]
        if len(true_indices) == 0: destory_id = 0
        else: destory_id = true_indices[0]
        repair_id = False
        destory_and_repair_id = False

    elif attribute_value == "table_r" and destory_id is not None and destory_id is not False:
         if not (0 <= destory_id < model.max_destroy_id):
              raise ValueError(f"Invalid destory_id ({destory_id}) passed to table_r selection.")

         rt_row_weight = model.rt_weight[destory_id]
         rt_row_weight_sum = sum(rt_row_weight)
         if rt_row_weight_sum <= 0 :
             model.rt_weight[destory_id] = np.ones_like(rt_row_weight) * 10
             rt_row_weight_sum = sum(model.rt_weight[destory_id])

         rt_cumsumprob = (rt_row_weight / rt_row_weight_sum).cumsum()
         rt_cumsumprob -= np.random.rand()
         true_indices = np.where(rt_cumsumprob > 0)[0]
         if len(true_indices) == 0: repair_id = 0
         else: repair_id = true_indices[0]
         destory_and_repair_id = False

    else:
        raise ValueError(f"Invalid attribute_value '{attribute_value}' or missing/invalid destory_id ({destory_id}) for table_r.")

    return destory_id if destory_id is not False else None, \
           repair_id if repair_id is not False else None, \
           destory_and_repair_id if destory_and_repair_id is not False else None


def doDestory(destory_id, model, sol):
    if destory_id == 0:
        reomve_list = createRandomDestory(model)
    elif destory_id == 1:
        reomve_list = createWorseDestory(model, sol)
    else:
        print(f"Warning: Invalid destory_id {destory_id} encountered. Using random destroy.")
        reomve_list = createRandomDestory(model)
    return reomve_list

def doRepair(repair_id, reomve_list, model, sol):
    if not reomve_list:
        return copy.deepcopy(sol)
    if repair_id == 0:
        new_sol = createRandomRepair(reomve_list, model, sol)
    elif repair_id == 1:
        new_sol = createGreedyRepair(reomve_list, model, sol)
    elif repair_id == 2:
        new_sol = createRegretRepair(reomve_list, model, sol)
    else:
         print(f"Warning: Invalid repair_id {repair_id} encountered. Using random repair.")
         new_sol = createRandomRepair(reomve_list, model, sol)
    return new_sol


def resetScore(model):
     model.d_score = np.zeros(model.max_destroy_id) # Use max_id for size
     model.r_score = np.zeros(model.max_repair_id)
     model.dr_score = np.zeros(model.max_destroy_id * model.max_repair_id)
     model.rt_score = np.zeros((model.max_destroy_id, model.max_repair_id))

def updateWeight(model, attribute_value, destory_id, repair_id):

    epsilon = 0.01
    if attribute_value in ["normal", "reward"]:
        if destory_id is not None and 0 <= destory_id < len(model.d_weight):
            model.d_weight[destory_id] = max(epsilon, model.rho * model.d_weight[destory_id] + (1 - model.rho) * model.d_score[destory_id])
        if repair_id is not None and 0 <= repair_id < len(model.r_weight):
            model.r_weight[repair_id] = max(epsilon, model.rho * model.r_weight[repair_id] + (1 - model.rho) * model.r_score[repair_id])

    elif attribute_value == "pair":
         if destory_id is not None and repair_id is not None:
            destory_and_repair_id = destory_id * model.max_repair_id + repair_id
            if 0 <= destory_and_repair_id < len(model.dr_weight):
                 model.dr_weight[destory_and_repair_id] = max(epsilon, model.rho * model.dr_weight[destory_and_repair_id] + (1 - model.rho) * model.dr_score[destory_and_repair_id])

    elif attribute_value == "table":
        if destory_id is not None and 0 <= destory_id < model.d_weight.shape[0]:
            model.d_weight[destory_id] = max(epsilon, model.rho * model.d_weight[destory_id] + (1 - model.rho) * model.d_score[destory_id])
        if destory_id is not None and repair_id is not None and \
           0 <= destory_id < model.rt_weight.shape[0] and \
           0 <= repair_id < model.rt_weight.shape[1]:
            model.rt_weight[destory_id][repair_id] = max(epsilon, model.rho * model.rt_weight[destory_id][repair_id] + (1 - model.rho) * model.rt_score[destory_id][repair_id])

def run_single_alns_instance_table(instance_number, params, seed, verbose):
    # 1. Set Random Seed for this process
    np.random.seed(seed)
    random.seed(seed)

    # 2. Create and Configure Model for this run
    model = Model()
    # ... (parameter setup as before) ...
    model.rand_d_max = params['rand_d_max']
    model.rand_d_min = params['rand_d_min']
    model.worst_d_min = params['worst_d_min']
    model.worst_d_max = params['worst_d_max']
    model.regret_n = params['regret_n']
    model.r1 = params['r1']
    model.r2 = params['r2']
    model.r3 = params['r3']
    model.r4 = params['r4']
    model.rho = params['rho']
    model.phi = params['phi']
    model.max_non_imp = params['max_non_imp']
    model.epochs = params['epochs']
    model.max_destroy_id = 2
    model.max_repair_id = 3

    # Initialize weights/scores for this run
    model.d_weight = np.ones(model.max_destroy_id) * 10
    model.d_score = np.zeros(model.max_destroy_id)
    model.r_weight = np.ones(model.max_repair_id) * 10
    model.r_score = np.zeros(model.max_repair_id)
    model.dr_weight = np.ones(model.max_destroy_id * model.max_repair_id) * 10
    model.dr_score = np.zeros(model.max_destroy_id * model.max_repair_id)
    model.rt_weight = np.ones((model.max_destroy_id, model.max_repair_id)) * 10
    model.rt_score = np.zeros((model.max_destroy_id, model.max_repair_id))


    # 3. Load Data and Calculate Distances
    try:
        readDateFile(instance_number, model)
        calDistanceMatrix(model)
    except Exception as e:
        print(f"Error loading data or calculating distances for instance {instance_number}, seed {seed}: {e}")
        return Sol(), []

    # 4. Initialize Solution
    sol = Sol()
    improvement_log = []
    if not model.node_id_list:
         print(f"Warning: No customer nodes found for instance {instance_number}, seed {seed}.")
         return Sol(), []

    try:
        sol.node_no_seq = genInitialSol(model.node_id_list)
        sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
        if sol.obj == float('inf'):
             print(f"Warning: Initial solution infeasible for seed {seed}.")
        model.best_sol = copy.deepcopy(sol)

        # Log the initial solution as the first "improvement"
        start_time = time.time()
        initial_log_entry = {
            "seed": seed,
            "epoch": -1,
            "time_elapsed_s": 0.0,
            "objective": model.best_sol.obj,
            # "routes": model.best_sol.route_list # Optional: adds detail but increases size
        }
        if model.best_sol.obj != float('inf'):
            improvement_log.append(initial_log_entry)

    except Exception as e:
        print(f"Error during initial solution generation for seed {seed}: {e}")
        return Sol(), [] # Return invalid solution


    # 5. ALNS Main Loop (adapted from run_iterate_table)
    num_non_imp = 0
    T = params.get('initial_T', 1000)
    state = True

    for ep in range(model.epochs):
        if (ep % 100 == 0):
            current_run_time = time.time() - start_time
            print(f"Epoch {ep} / {model.epochs} (Seed {seed}) - Time: {current_run_time:.2f}s")
        resetScore(model)

        try:
             destory_id, _, _ = selectDestoryRepair(model, "table_d")
             if destory_id is None:
                 print(f"Warning (Seed {seed}, Epoch {ep}): Failed to select destroy operator. Skipping.")
                 continue

             _, repair_id, _ = selectDestoryRepair(model, "table_r", destory_id=destory_id)
             if repair_id is None:
                 print(f"Warning (Seed {seed}, Epoch {ep}): Failed to select repair operator for destroy_id {destory_id}. Skipping.")
                 continue

        except (IndexError, ValueError) as e_select:
             print(f"Warning (Seed {seed}, Epoch {ep}): Operator selection failed ({e_select}). Resetting weights.")
             model.d_weight = np.ones(model.max_destroy_id) * 10
             model.rt_weight = np.ones((model.max_destroy_id, model.max_repair_id)) * 10
             continue
        except Exception as e_select_other:
             print(f"Unexpected error during operator selection (Seed {seed}, Epoch {ep}): {e_select_other}. Skipping.")
             continue

        try:
            reomve_list = doDestory(destory_id, model, sol)
            new_sol = doRepair(repair_id, reomve_list, model, sol)
            if new_sol.obj == float('inf'):
                pass
        except Exception as e_op:
            print(f"Error during destroy/repair (Seed {seed}, Epoch {ep}): {e_op}. Skipping.")
            continue


        # Acceptance Criteria
        accepted = False
        improved_global = False


        if new_sol.obj < sol.obj:
            sol = copy.deepcopy(new_sol)
            accepted = True
            model.d_score[destory_id] += model.r2
            model.rt_score[destory_id][repair_id] += model.r2

            if new_sol.obj < model.best_sol.obj:
                num_non_imp = 0
                model.best_sol = copy.deepcopy(new_sol)
                improved_global = True
                model.d_score[destory_id] += model.r1
                model.rt_score[destory_id][repair_id] += model.r1
                current_time = time.time() - start_time
                log_entry = {
                    "seed": seed,
                    "epoch": ep,
                    "time_elapsed_s": round(current_time, 4),
                    "objective": model.best_sol.obj,
                     # "node_sequence": model.best_sol.node_no_seq,
                     # "routes": copy.deepcopy(model.best_sol.route_list) # Optional
                }
                improvement_log.append(log_entry)
            else:
                 num_non_imp += 1

        # SA Acceptance for non-improving solutions
        elif new_sol.obj != float('inf') and math.exp((sol.obj - new_sol.obj) / max(T, 0.001)) >= random.random():
            num_non_imp += 1
            sol = copy.deepcopy(new_sol)
            model.d_score[destory_id] += model.r3
            model.rt_score[destory_id][repair_id] += model.r3
            accepted = True
        else:
            num_non_imp += 1
            model.d_score[destory_id] += model.r4
            model.rt_score[destory_id][repair_id] += model.r4

        T = T * model.phi
        updateWeight(model, "table", destory_id, repair_id)

        if num_non_imp >= model.max_non_imp:
             if state and params.get('dynamic_restart', False):
                 model.max_non_imp += ep
                 state = False

             try:
                 sol.node_no_seq = genInitialSol(model.node_id_list)
                 sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
                 T = params.get('initial_T', 1000) # Reset temperature
                 num_non_imp = 0
                 if verbose: print(f"Seed {seed}: Restarting at epoch {ep} (New sol obj: {sol.obj:.2f})")
                 if sol.obj < model.best_sol.obj:
                      model.best_sol = copy.deepcopy(sol)
                      current_time = time.time() - start_time
                      log_entry = {
                          "seed": seed,
                          "epoch": ep,
                          "time_elapsed_s": round(current_time, 4),
                          "objective": model.best_sol.obj,
                          # "node_sequence": model.best_sol.node_no_seq,
                      }
                      improvement_log.append(log_entry)
                      if verbose: print(f"Seed {seed} Ep {ep}: New best obj after restart: {model.best_sol.obj:.2f}")

             except Exception as e_restart:
                 print(f"Error during restart (Seed {seed}, Epoch {ep}): {e_restart}. Continuing loop.")
                 num_non_imp = 0


    run_time = time.time() - start_time
    if verbose:
        final_obj = model.best_sol.obj if model.best_sol else float('inf')
        print(f"Run with seed {seed} finished. Best Obj: {final_obj:.2f}, Time: {run_time:.2f}s")

    if model.best_sol is None: model.best_sol = Sol()
    return model.best_sol, improvement_log


def run_single_alns_instance_pair(instance_number, params, seed, verbose):

    np.random.seed(seed)
    random.seed(seed)
    model = Model()
    model.rand_d_max = params['rand_d_max']
    model.rand_d_min = params['rand_d_min']
    model.worst_d_min = params['worst_d_min']
    model.worst_d_max = params['worst_d_max']
    model.regret_n = params['regret_n']
    model.r1 = params['r1']
    model.r2 = params['r2']
    model.r3 = params['r3']
    model.r4 = params['r4']
    model.rho = params['rho']
    model.phi = params['phi']
    model.max_non_imp = params['max_non_imp']
    model.epochs = params['epochs']
    model.max_destroy_id = 2
    model.max_repair_id = 3

    model.d_weight = np.ones(model.max_destroy_id) * 10
    model.d_score = np.zeros(model.max_destroy_id)
    model.r_weight = np.ones(model.max_repair_id) * 10
    model.r_score = np.zeros(model.max_repair_id)
    model.dr_weight = np.ones(model.max_destroy_id * model.max_repair_id) * 10
    model.dr_score = np.zeros(model.max_destroy_id * model.max_repair_id)
    model.rt_weight = np.ones((model.max_destroy_id, model.max_repair_id)) * 10
    model.rt_score = np.zeros((model.max_destroy_id, model.max_repair_id))

    try:
        readDateFile(instance_number, model)
        calDistanceMatrix(model)
    except Exception as e:
        print(f"Error loading data or calculating distances for instance {instance_number}, seed {seed}: {e}")
        return Sol(), []

    sol = Sol()
    improvement_log = []
    if not model.node_id_list:
        print(f"Warning: No customer nodes found for instance {instance_number}, seed {seed}.")
        return Sol(), []

    try:
        sol.node_no_seq = genInitialSol(model.node_id_list)
        sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
        if sol.obj == float('inf'):
            print(f"Warning: Initial solution infeasible for seed {seed}.")
        model.best_sol = copy.deepcopy(sol)

        start_time = time.time()
        initial_log_entry = {
            "seed": seed,
            "epoch": -1,
            "time_elapsed_s": 0.0,
            "objective": model.best_sol.obj,
        }
        if model.best_sol.obj != float('inf'):
            improvement_log.append(initial_log_entry)

    except Exception as e:
        print(f"Error during initial solution generation for seed {seed}: {e}")
        return Sol(), []

    num_non_imp = 0
    T = params.get('initial_T', 1000)
    state = True

    for ep in range(model.epochs):
        if (ep % 100 == 0):
            current_run_time = time.time() - start_time
            print(f"Epoch {ep} / {model.epochs} (Seed {seed}) - Time: {current_run_time:.2f}s")

        resetScore(model)

        try:
            destory_id, repair_id, destroy_and_repair_id = selectDestoryRepair(model, "pair")
            if destory_id is None or repair_id is None or destroy_and_repair_id is None:
                print(f"Warning (Seed {seed}, Epoch {ep}): Failed to select operator pair. Skipping.")
                continue

        except (IndexError, ValueError) as e_select:
            print(f"Warning (Seed {seed}, Epoch {ep}): Operator selection failed ({e_select}). Resetting weights.")
            model.dr_weight = np.ones(model.max_destroy_id * model.max_repair_id) * 10
            continue
        except Exception as e_select_other:
            print(f"Unexpected error during operator selection (Seed {seed}, Epoch {ep}): {e_select_other}. Skipping.")
            continue

        try:
            reomve_list = doDestory(destory_id, model, sol)
            new_sol = doRepair(repair_id, reomve_list, model, sol)
        except Exception as e_op:
            print(f"Error during destroy/repair (Seed {seed}, Epoch {ep}): {e_op}. Skipping.")
            continue

        if new_sol.obj < sol.obj:
            sol = copy.deepcopy(new_sol)
            if new_sol.obj < model.best_sol.obj:
                num_non_imp = 0
                model.best_sol = copy.deepcopy(new_sol)
                model.dr_score[destroy_and_repair_id] += model.r1

                current_time = time.time() - start_time
                log_entry = {
                    "seed": seed,
                    "epoch": ep,
                    "time_elapsed_s": round(current_time, 4),
                    "objective": model.best_sol.obj,
                }
                improvement_log.append(log_entry)

            else:
                num_non_imp += 1
                model.dr_score[destroy_and_repair_id] += model.r2

        elif new_sol.obj != float('inf') and math.exp((sol.obj - new_sol.obj) / max(T, 0.001)) >= random.random():
            num_non_imp += 1
            sol = copy.deepcopy(new_sol)
            model.dr_score[destroy_and_repair_id] += model.r3
        else:
            num_non_imp += 1
            model.dr_score[destroy_and_repair_id] += model.r4

        T = T * model.phi
        updateWeight(model, "pair", destory_id, repair_id)

        if num_non_imp > model.max_non_imp:
            if state:
                model.max_non_imp += ep
                state = False

            try:
                sol.node_no_seq = genInitialSol(model.node_id_list)
                sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
                T = params.get('initial_T', 1000)
                num_non_imp = 0
                if verbose: print(f"Seed {seed}: Restarting at epoch {ep} (New sol obj: {sol.obj:.2f})")

                if sol.obj < model.best_sol.obj:
                    model.best_sol = copy.deepcopy(sol)
                    current_time = time.time() - start_time
                    log_entry = {
                        "seed": seed,
                        "epoch": ep,
                        "time_elapsed_s": round(current_time, 4),
                        "objective": model.best_sol.obj,
                    }
                    improvement_log.append(log_entry)
                    if verbose: print(f"Seed {seed} Ep {ep}: New best obj after restart: {model.best_sol.obj:.2f}")

            except Exception as e_restart:
                print(f"Error during restart (Seed {seed}, Epoch {ep}): {e_restart}. Continuing loop.")
                num_non_imp = 0

    run_time = time.time() - start_time
    if verbose:
        final_obj = model.best_sol.obj if model.best_sol else float('inf')
        print(f"Run with seed {seed} finished. Best Obj: {final_obj:.2f}, Time: {run_time:.2f}s")

    if model.best_sol is None: model.best_sol = Sol()
    return model.best_sol, improvement_log


def run_single_alns_instance_normal(instance_number, params, seed, verbose):
    # 1. Set Random Seed for this process
    np.random.seed(seed)
    random.seed(seed)

    # 2. Create and Configure Model for this run
    model = Model()
    model.rand_d_max = params['rand_d_max']
    model.rand_d_min = params['rand_d_min']
    model.worst_d_min = params['worst_d_min']
    model.worst_d_max = params['worst_d_max']
    model.regret_n = params['regret_n']
    model.r1 = params['r1']
    model.r2 = params['r2']
    model.r3 = params['r3']
    model.r4 = params['r4']
    model.rho = params['rho']
    model.phi = params['phi']
    model.max_non_imp = params['max_non_imp']
    model.epochs = params['epochs']
    model.max_destroy_id = 2
    model.max_repair_id = 3

    # Initialize weights/scores for this run
    model.d_weight = np.ones(model.max_destroy_id) * 10
    model.d_score = np.zeros(model.max_destroy_id)
    model.r_weight = np.ones(model.max_repair_id) * 10
    model.r_score = np.zeros(model.max_repair_id)
    model.dr_weight = np.ones(model.max_destroy_id * model.max_repair_id) * 10
    model.dr_score = np.zeros(model.max_destroy_id * model.max_repair_id)
    model.rt_weight = np.ones((model.max_destroy_id, model.max_repair_id)) * 10
    model.rt_score = np.zeros((model.max_destroy_id, model.max_repair_id))

    # 3. Load Data and Calculate Distances
    try:
        readDateFile(instance_number, model)
        calDistanceMatrix(model)
    except Exception as e:
        print(f"Error loading data or calculating distances for instance {instance_number}, seed {seed}: {e}")
        return Sol(), []

    # 4. Initialize Solution
    sol = Sol()
    improvement_log = []
    if not model.node_id_list:
        print(f"Warning: No customer nodes found for instance {instance_number}, seed {seed}.")
        return Sol(), []

    try:
        sol.node_no_seq = genInitialSol(model.node_id_list)
        sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
        if sol.obj == float('inf'):
            print(f"Warning: Initial solution infeasible for seed {seed}.")
        model.best_sol = copy.deepcopy(sol)

        # Log the initial solution as the first "improvement"
        start_time = time.time()
        initial_log_entry = {
            "seed": seed,
            "epoch": -1,
            "time_elapsed_s": 0.0,
            "objective": model.best_sol.obj,
        }
        if model.best_sol.obj != float('inf'):
            improvement_log.append(initial_log_entry)

    except Exception as e:
        print(f"Error during initial solution generation for seed {seed}: {e}")
        return Sol(), []

    # 5. ALNS Main Loop (using "normal" mode)
    num_non_imp = 0
    T = params.get('initial_T', 1000)
    state = True

    for ep in range(model.epochs):
        if (ep % 100 == 0):
            current_run_time = time.time() - start_time
            print(f"Epoch {ep} / {model.epochs} (Seed {seed}) - Time: {current_run_time:.2f}s")

        resetScore(model)

        try:
            destory_id, repair_id, _ = selectDestoryRepair(model, "normal")
            if destory_id is None or repair_id is None:
                print(f"Warning (Seed {seed}, Epoch {ep}): Failed to select operators. Skipping.")
                continue

        except (IndexError, ValueError) as e_select:
            print(f"Warning (Seed {seed}, Epoch {ep}): Operator selection failed ({e_select}). Resetting weights.")
            model.d_weight = np.ones(model.max_destroy_id) * 10
            model.r_weight = np.ones(model.max_repair_id) * 10
            continue
        except Exception as e_select_other:
            print(f"Unexpected error during operator selection (Seed {seed}, Epoch {ep}): {e_select_other}. Skipping.")
            continue

        # Perform destroy and repair
        try:
            reomve_list = doDestory(destory_id, model, sol)
            new_sol = doRepair(repair_id, reomve_list, model, sol)
        except Exception as e_op:
            print(f"Error during destroy/repair (Seed {seed}, Epoch {ep}): {e_op}. Skipping.")
            continue

        if new_sol.obj < sol.obj:
            sol = copy.deepcopy(new_sol)
            if new_sol.obj < model.best_sol.obj:
                num_non_imp = 0
                model.best_sol = copy.deepcopy(new_sol)
                model.d_score[destory_id] += model.r1
                model.r_score[repair_id] += model.r1

                # Log the improvement
                current_time = time.time() - start_time
                log_entry = {
                    "seed": seed,
                    "epoch": ep,
                    "time_elapsed_s": round(current_time, 4),
                    "objective": model.best_sol.obj,
                }
                improvement_log.append(log_entry)

            else:
                num_non_imp += 1
                model.d_score[destory_id] += model.r2
                model.r_score[repair_id] += model.r2

        elif new_sol.obj != float('inf') and math.exp((sol.obj - new_sol.obj) / max(T, 0.001)) >= random.random():
            num_non_imp += 1
            sol = copy.deepcopy(new_sol)
            model.d_score[destory_id] += model.r3
            model.r_score[repair_id] += model.r3
        else:
            num_non_imp += 1
            model.d_score[destory_id] += model.r4
            model.r_score[repair_id] += model.r4

        # Update temperature and weights
        T = T * model.phi
        updateWeight(model, "normal", destory_id, repair_id)

        # Restart Logic
        if num_non_imp > model.max_non_imp:
            if state:
                model.max_non_imp += ep
                state = False

            try:
                sol.node_no_seq = genInitialSol(model.node_id_list)
                sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
                T = params.get('initial_T', 1000)
                num_non_imp = 0
                if verbose: print(f"Seed {seed}: Restarting at epoch {ep} (New sol obj: {sol.obj:.2f})")

                if sol.obj < model.best_sol.obj:
                    model.best_sol = copy.deepcopy(sol)
                    current_time = time.time() - start_time
                    log_entry = {
                        "seed": seed,
                        "epoch": ep,
                        "time_elapsed_s": round(current_time, 4),
                        "objective": model.best_sol.obj,
                    }
                    improvement_log.append(log_entry)
                    if verbose: print(f"Seed {seed} Ep {ep}: New best obj after restart: {model.best_sol.obj:.2f}")

            except Exception as e_restart:
                print(f"Error during restart (Seed {seed}, Epoch {ep}): {e_restart}. Continuing loop.")
                num_non_imp = 0

    run_time = time.time() - start_time
    if verbose:
        final_obj = model.best_sol.obj if model.best_sol else float('inf')
        print(f"Run with seed {seed} finished. Best Obj: {final_obj:.2f}, Time: {run_time:.2f}s")

    if model.best_sol is None: model.best_sol = Sol()
    return model.best_sol, improvement_log


def run_single_alns_instance_reward(instance_number, params, seed, verbose):
    # 1. Set Random Seed for this process
    np.random.seed(seed)
    random.seed(seed)

    # 2. Create and Configure Model for this run
    model = Model()
    model.rand_d_max = params['rand_d_max']
    model.rand_d_min = params['rand_d_min']
    model.worst_d_min = params['worst_d_min']
    model.worst_d_max = params['worst_d_max']
    model.regret_n = params['regret_n']
    model.r1 = params['r1']
    model.r2 = params['r2']
    model.r3 = params['r3']
    model.r4 = params['r4']
    model.rho = params['rho']
    model.phi = params['phi']
    model.max_non_imp = params['max_non_imp']
    model.epochs = params['epochs']
    model.max_destroy_id = 2
    model.max_repair_id = 3

    # Initialize weights/scores for this run
    model.d_weight = np.ones(model.max_destroy_id) * 10
    model.d_score = np.zeros(model.max_destroy_id)
    model.r_weight = np.ones(model.max_repair_id) * 10
    model.r_score = np.zeros(model.max_repair_id)
    model.dr_weight = np.ones(model.max_destroy_id * model.max_repair_id) * 10
    model.dr_score = np.zeros(model.max_destroy_id * model.max_repair_id)
    model.rt_weight = np.ones((model.max_destroy_id, model.max_repair_id)) * 10
    model.rt_score = np.zeros((model.max_destroy_id, model.max_repair_id))

    # 3. Load Data and Calculate Distances
    try:
        readDateFile(instance_number, model)
        calDistanceMatrix(model)
    except Exception as e:
        print(f"Error loading data or calculating distances for instance {instance_number}, seed {seed}: {e}")
        return Sol(), []

    # 4. Initialize Solution
    sol = Sol()
    improvement_log = []
    if not model.node_id_list:
        print(f"Warning: No customer nodes found for instance {instance_number}, seed {seed}.")
        return Sol(), []

    try:
        sol.node_no_seq = genInitialSol(model.node_id_list)
        sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
        if sol.obj == float('inf'):
            print(f"Warning: Initial solution infeasible for seed {seed}.")
        model.best_sol = copy.deepcopy(sol)

        # Log the initial solution as the first "improvement"
        start_time = time.time()
        initial_log_entry = {
            "seed": seed,
            "epoch": -1,
            "time_elapsed_s": 0.0,
            "objective": model.best_sol.obj,
        }
        if model.best_sol.obj != float('inf'):
            improvement_log.append(initial_log_entry)

    except Exception as e:
        print(f"Error during initial solution generation for seed {seed}: {e}")
        return Sol(), []

    # 5. ALNS Main Loop (using "reward" mode with dynamic r1)
    num_non_imp = 0
    T = params.get('initial_T', 1000)
    setting_dif = sol.obj / 100
    large_delta = 100
    w_fix = 3
    state = True

    for ep in range(model.epochs):
        if (ep % 100 == 0):
            current_run_time = time.time() - start_time
            print(f"Epoch {ep} / {model.epochs} (Seed {seed}) - Time: {current_run_time:.2f}s")

        resetScore(model)

        try:
            destory_id, repair_id, _ = selectDestoryRepair(model, "reward")
            if destory_id is None or repair_id is None:
                print(f"Warning (Seed {seed}, Epoch {ep}): Failed to select operators. Skipping.")
                continue

        except (IndexError, ValueError) as e_select:
            print(f"Warning (Seed {seed}, Epoch {ep}): Operator selection failed ({e_select}). Resetting weights.")
            model.d_weight = np.ones(model.max_destroy_id) * 10
            model.r_weight = np.ones(model.max_repair_id) * 10
            continue
        except Exception as e_select_other:
            print(f"Unexpected error during operator selection (Seed {seed}, Epoch {ep}): {e_select_other}. Skipping.")
            continue

        # Perform destroy and repair
        try:
            reomve_list = doDestory(destory_id, model, sol)
            new_sol = doRepair(repair_id, reomve_list, model, sol)
        except Exception as e_op:
            print(f"Error during destroy/repair (Seed {seed}, Epoch {ep}): {e_op}. Skipping.")
            continue

        if new_sol.obj < sol.obj:
            sol = copy.deepcopy(new_sol)
            if new_sol.obj < model.best_sol.obj:
                target_dif = model.best_sol.obj - new_sol.obj
                model.r1 = max(w_fix, min(0.1 * large_delta,
                                          w_fix + round(large_delta * num_non_imp / model.max_non_imp) + round(
                                              target_dif / setting_dif)))

                num_non_imp = 0
                model.best_sol = copy.deepcopy(new_sol)
                model.d_score[destory_id] += model.r1
                model.r_score[repair_id] += model.r1

                # Log the improvement
                current_time = time.time() - start_time
                log_entry = {
                    "seed": seed,
                    "epoch": ep,
                    "time_elapsed_s": round(current_time, 4),
                    "objective": model.best_sol.obj,
                }
                improvement_log.append(log_entry)

            else:
                num_non_imp += 1
                model.d_score[destory_id] += model.r2
                model.r_score[repair_id] += model.r2

        elif new_sol.obj != float('inf') and math.exp((sol.obj - new_sol.obj) / max(T, 0.001)) >= random.random():
            num_non_imp += 1
            sol = copy.deepcopy(new_sol)
            model.d_score[destory_id] += model.r3
            model.r_score[repair_id] += model.r3
        else:
            num_non_imp += 1
            model.d_score[destory_id] += model.r4
            model.r_score[repair_id] += model.r4

        # Update temperature and weights
        T = T * model.phi
        updateWeight(model, "reward", destory_id, repair_id)

        # Restart Logic
        if num_non_imp > model.max_non_imp:
            if state:
                model.max_non_imp += ep
                state = False

            try:
                sol.node_no_seq = genInitialSol(model.node_id_list)
                sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
                T = params.get('initial_T', 1000)
                num_non_imp = 0
                if verbose: print(f"Seed {seed}: Restarting at epoch {ep} (New sol obj: {sol.obj:.2f})")

                if sol.obj < model.best_sol.obj:
                    model.best_sol = copy.deepcopy(sol)
                    current_time = time.time() - start_time
                    log_entry = {
                        "seed": seed,
                        "epoch": ep,
                        "time_elapsed_s": round(current_time, 4),
                        "objective": model.best_sol.obj,
                    }
                    improvement_log.append(log_entry)
                    if verbose: print(f"Seed {seed} Ep {ep}: New best obj after restart: {model.best_sol.obj:.2f}")

            except Exception as e_restart:
                print(f"Error during restart (Seed {seed}, Epoch {ep}): {e_restart}. Continuing loop.")
                num_non_imp = 0

    run_time = time.time() - start_time
    if verbose:
        final_obj = model.best_sol.obj if model.best_sol else float('inf')
        print(f"Run with seed {seed} finished. Best Obj: {final_obj:.2f}, Time: {run_time:.2f}s")

    if model.best_sol is None: model.best_sol = Sol()
    return model.best_sol, improvement_log


# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Good practice

    # --- Configuration ---
    instance_number = 5
    num_parallel_runs = 2
    base_random_seed = 10000
    output_json_filename = f"alns_instance_{instance_number}_improvements_log_reward.json"


    # ALNS Parameters Dictionary
    alns_params = {
        'rand_d_max': 0.3,
        'rand_d_min': 0.1,
        'worst_d_min': 5,
        'worst_d_max': 20,
        'regret_n': 3,
        'r1': 3, 'r2': 2, 'r3': 1, 'r4': 0,
        'rho': 0.9,
        'phi': 0.95,
        'max_non_imp': 300,
        'epochs': 2,
        'initial_T': 1000,
        'dynamic_restart': False
    }
    verbose_per_run = False

    # --- Prepare tasks for parallel execution ---
    tasks = []
    for i in range(num_parallel_runs):
        run_seed = base_random_seed + i * 10 # Generate unique seeds
        tasks.append((instance_number, alns_params, run_seed, verbose_per_run))

    print(f"Starting {num_parallel_runs} parallel ALNS runs for instance {instance_number}...")
    overall_start_time = time.time()

    # --- Execute in Parallel ---
    all_improvement_logs = []
    results_list = []
    try:
        with multiprocessing.Pool(processes=num_parallel_runs) as pool:
            # Use starmap to pass multiple arguments unpacked from tuples
            # results_list = pool.starmap(run_single_alns_instance_table, tasks)
            # results_list = pool.starmap(run_single_alns_instance_pair, tasks)
            # results_list = pool.starmap(run_single_alns_instance_normal, tasks)
            results_list = pool.starmap(run_single_alns_instance_reward, tasks)
    except Exception as e_pool:
         print(f"Error during multiprocessing execution: {e_pool}")


    # --- Process Results ---
    overall_best_sol = Sol()  # Initialize with infinite objective
    valid_objs = []
    for result_tuple in results_list:
        if result_tuple is None: # Handle potential None return from error
             continue
        sol, log_list = result_tuple # Unpack the tuple

        # Process the solution
        if sol is not None and sol.obj < float('inf'):
            valid_objs.append(sol.obj)
            if sol.obj < overall_best_sol.obj:
                overall_best_sol = sol # Keep track of the best solution object

        # Collect the logs
        if log_list: # Check if the log list is not empty
            all_improvement_logs.extend(log_list) # Add logs from this run

    # --- Calculate Average Objective ---
    if valid_objs:
        avg_obj = sum(valid_objs) / len(valid_objs)
    else:
        avg_obj = float('inf')

    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time

    # --- Sort and Save Improvement Logs to JSON ---
    if all_improvement_logs:
        # Sort logs: first by seed, then by epoch within each seed
        all_improvement_logs.sort(key=lambda x: (x['seed'], x['epoch']))
        print(f"\nSaving {len(all_improvement_logs)} improvement log entries...")
        try:
            with open(output_json_filename, 'w') as f:
                json.dump(all_improvement_logs, f, indent=2) # Use indent for readability
            print(f"Improvement logs saved to {output_json_filename}")
        except IOError as e:
            print(f"Error writing JSON log file: {e}")
        except TypeError as e:
             # This might happen if non-serializable types (like numpy arrays) are in the log
             print(f"Error serializing data to JSON (check data types in log_entry): {e}")
             # You might need to convert numpy types using .tolist() before adding to log_entry
    else:
        print("\nNo improvement log entries were generated.")


    # --- Output Final Best Result and Average ---
    print("\n--- Overall Best Result ---")
    if overall_best_sol.obj == float('inf'):
        print("No feasible solution found across all runs.")
    else:
        print(f"Best Objective Found: {overall_best_sol.obj:.2f}")
        print("Best Solution Routes:")
        if overall_best_sol.route_list:
            for i, route in enumerate(overall_best_sol.route_list):
                # Ensure route is printable (list of IDs)
                route_str = ', '.join(map(str, route))
                print(f"  Route {i+1}: {route_str}")
        else:
            print("  (Route list not available or solution infeasible)")

    # Print average solution
    if avg_obj == float('inf'):
        print("Average Objective: No feasible solutions to average.")
    else:
        print(f"Average Objective over {len(valid_objs)} valid runs: {avg_obj:.2f}")

    print(f"\nTotal execution time ({num_parallel_runs} processes): {total_runtime:.2f} seconds")
