import pickle
import os
import collections
import numpy as np
import math
import copy
import torch
import torch.multiprocessing as mp
import datetime
import logging

from alpha_net import AlphaZeroNet
import encode_decode_board as ed
from board import Board


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


class UCTNode():
    def __init__(self, state, action, parent=None):
        """
        Upper confidence tree node
        """
        self.state = state # state s
        self.action = action # action index
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.action_indxes = []

    @property
    def num_visits(self):
        return self.parent.child_number_visits[action]

    @num_visits.setter
    def num_visits(self, value):
        self.parent.child_total_value[self.action] = value
    
    @property
    def total_value(self, ):
        self.parent.child_total_value[self.action] = value

    
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current_board.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                          dtype=np.float32)+192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def decode_n_move_pieces(self,board,move):
        board.drop_piece(move)
        return board

    def expand(self, child_priors):
        self.is_expanded = True 
        action_idxs = self.state.actions()
        c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        # Mask all illegal actions
        for i in range(len(child_priors)):
            if i not in child_priors:
                c_p[i] = 0.0

        # add dirichlet noise to child_priors in root node
        if self.parent.parent == None:
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.num_visits += 1
            if current.state.player == 1:
                current.total_value += (1*value_estimate)
            elif current.state.player == 0:
                current.total_value += ( -1*value_estimate)
            current = current.parent
    
    
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

    
def UCT_search(game_state, num_reads, net, temp):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.state)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        if leaf.game.check_winner() == True or leaf.game.actions() == []: # if somebody won or draw
            leaf.backup(value_estimate); continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)
    return root


def get_policy(root, temperature):
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))


def do_decode_n_move_pieces(board,move):
    board.drop_piece(move)
    return board

def MCTS_self_play(net, num_games, start_idx, cpu, args, iteration):
    logger.info("[CPU: %d]: Starting MCTS self-play..." % cpu)
    for idx in range(start_idx, num_games + start_idx):
        current_board = Board()
        checkmate = False
        dataset = [] # store state, policy value for neural network training
        states = []; value = 0; move_count = 0
        while checkmate == False and current_board.actions != []:
            if move_count < 11:
                temperature = args.temperature_MCTS
            else:
                temperature = 0.1
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            root = UCT_search(current_board, 777, net, temperature)
            policy = get_policy(root, temperature)
            print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idx), policy)
            current_board = do_decode_n_move_pieces(current_board, 
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy) )
            
            dataset.append([board_state,policy])
            print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (iteration, cpu, idxx), current_board.current_board,current_board.player); print(" ")
            if current_board.check_winner() == True: # if somebody won
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
            move_count += 1

        dataset_p = []
        for index, data in enumerate(dataset):
            state, policy = data
            if index == 0:
                dataset_p.append([state, policy, 0])
            else:
                dataset_p.append([state, policy, value])
        del dataset
        save_as_pickle("iter_%d/" % iteration +\
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)

def run_MCTS(args, start_idx, iteration):
    net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    net = AlphaZeroNet()
    net.cuda()
    if args.MCTS_num_processes > 1:
        logger.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn",force=True)
        net.share_memory()
        net.eval()
    
        current_net_filename = os.path.join("./model_data/",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                        net_to_play))
            logger.info("Initialized model.")
        
        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes
        
        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=MCTS_self_play, args=(net, args.num_games_per_MCTS_process, start_idx, i, args, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("Finished multi-process MCTS!")
    
    elif args.MCTS_num_processes == 1:
        logger.info("Preparing model for MCTS...")
        net.eval()
        
        current_net_filename = os.path.join("./model_data/",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                        net_to_play))
            logger.info("Initialized model.")
        
        with torch.no_grad():
            MCTS_self_play(net, args.num_games_per_MCTS_process, start_idx, 0, args, iteration)
        logger.info("Finished MCTS!")