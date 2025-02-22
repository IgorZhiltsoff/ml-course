import string
import os
from random import sample

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
from copy import deepcopy
import json

import matplotlib.pyplot as plt

class SimplestRNN:
    def __init__(self,
                tokens, # SHOULD END WITH SOS AND EOS
                neurons=100
    ):
        self.tokens = tokens
        self.num_tokens = len(tokens)
        
        self.token_to_idx = {x: idx for idx, x in enumerate(tokens)}
        self.idx_to_token = {idx: x for idx, x in enumerate(tokens)}

        self.header_token = tokens[-2]
        self.footer_token = tokens[-1]
        self.header_idx = self.token_to_idx[self.header_token]
        self.footer_idx = self.token_to_idx[self.footer_token]


        # hyperparameters

        # parameters
        self.in_dim = self.num_tokens
        self.hid_dim = neurons # hidden state is NOT (log) proba. Output (and input) is
        self.out_dim = self.num_tokens

        self.in2hid = np.random.normal(size=(self.hid_dim, self.in_dim))
        self.hid2hid = np.random.normal(size=(self.hid_dim, self.hid_dim))
        self.hid2out = np.random.normal(size=(self.out_dim, self.hid_dim))

        self.bias_before_activation = np.random.normal(size=self.hid_dim)
        self.out_logproba_bias = np.random.normal(size=self.out_dim)

        # internal
        self.hid = np.zeros(self.hid_dim) # self.hid = np.random.normal(size=self.hid_dim)

        # history
        self.loss_history = []
        self.hid_history = [] # for visualizing activation
    
    # FIT 
    def fit(self, human_written_text, 
            line_count=256, line_length=100,
            iterations=1,
            learning_rate=0.3, l2_reg=0.0, clip=None,):

        text_encoded = self.encode(human_written_text)
        for it in range(iterations):
            inputs = self.make_chunks(text_encoded, line_count=line_count, line_length=line_length)

            targets = [x[1:] for x in inputs]
            inputs = [x[:-1] for x in inputs]

            self.fit_with_chunk(inputs, targets, 
                                        learning_rate=learning_rate, l2_reg=l2_reg, clip=clip)

        # history
        self.learning_rate = learning_rate
        self.line_count = line_count
        self.line_length = line_length
        self.l2_reg = l2_reg
        self.clip = clip


    def fit_with_chunk(self, 
            inputs,
            targets,
            learning_rate, l2_reg, clip
           ):
        
        for it in range(len(inputs)):
            self.reset_hid() # TODO should I?
            input_line = inputs[it] 
            target_line = targets[it] 
            loss = 0

            n = len(input_line)

            # history for back propagation
            out_proba_at = [None]
            in_proba_at = [None]
            hid_at = [np.copy(self.hid)]
            out_logproba_at = [None] # -1-st element for back propagation

            for t in range(n):
                out_proba, (in_proba, self.hid, out_logproba,) = self.predict_proba(input_line[t])

                out_proba_at.append(np.copy(out_proba))
                in_proba_at.append(np.copy(in_proba))
                hid_at.append(np.copy(self.hid))
                out_logproba_at.append(np.copy(out_logproba))

                loss += -np.log(out_proba[target_line[t]])

                    
            # make gradient step 
            in2hid_delta = np.zeros_like(self.in2hid)
            hid2hid_delta = np.zeros_like(self.hid2hid)
            hid2out_delta = np.zeros_like(self.hid2out)

            bias_before_activation_delta = np.zeros_like(self.bias_before_activation)
            out_logproba_bias_delta = np.zeros_like(self.out_logproba_bias)

            param_changes = [
                (self.in2hid, in2hid_delta),
                (self.hid2hid, hid2hid_delta),
                (self.hid2out, hid2out_delta),

                (self.bias_before_activation, bias_before_activation_delta),
                (self.out_logproba_bias, out_logproba_bias_delta),
            ]

            dloss_d_hid_at_t_plus_1 = np.zeros_like(self.hid)

            # back propagation, that is:
            # we want to know grad_param loss
            # But loss depends on params via y:
            # params --f--> y --g--> loss
            # Then grad_params loss = (D_params f)* grad_y loss
            # BECAUSE GRADIENT IS A COvector FIELD!
            for t in reversed(range(1, n+1)):
                # ALL OF THIS COMES FROM UNFOLDING COMPUTATION DIAGRAM 
                # https://www.deeplearningbook.org/contents/rnn.html, 10.2.2

                dloss_d_out_logproba = out_proba_at[t] 
                dloss_d_out_logproba[target_line[t - 1]] -= 1
                # EXCEPT FOR THE TWO LINES ABOVE, NOTHING DEPENDS ON THE CHOICE OF LOSS FUNCTION

                dloss_d_hid_at_t = self.hid2out.T @ dloss_d_out_logproba + dloss_d_hid_at_t_plus_1 # recursive

                activation_derivative_at_hid_t = (1 - hid_at[t] * hid_at[t])
                dloss_d_preactivated_hid_at_t = activation_derivative_at_hid_t * dloss_d_hid_at_t # just (transposed) diagonal matrix

                # param_delta = dloss_dparam, thus the sum
                in2hid_delta += self.outer_vector_product(dloss_d_preactivated_hid_at_t, in_proba_at[t]) # no recursion

                hid2hid_delta += self.outer_vector_product(dloss_d_preactivated_hid_at_t, hid_at[t-1])

                hid2out_delta += self.outer_vector_product(dloss_d_out_logproba, hid_at[t])

                bias_before_activation_delta += dloss_d_preactivated_hid_at_t # recursive
                out_logproba_bias_delta += dloss_d_out_logproba # no recursion

                # WOULD HAVE BEEN MORE INTUITIVE TO PLACE THIS AT THE START OF THE LOOP
                # BUT BREAKS THE FIRST ITERATION :(
                dloss_d_hid_at_t_plus_1 = self.hid2hid.T @ activation_derivative_at_hid_t * dloss_d_preactivated_hid_at_t

            # update params
            for param, param_delta in param_changes:
                if clip:
                    param_delta = np.clip(param_delta, -clip, clip) 
                param += -learning_rate * param_delta

            if it % 100 == 0:
                self.loss_history.append(loss)
                print(f'Loss on iteration {it} is {loss}')
                print(self.scales_of_change(param_changes))

    # PREDICT
    def predict_proba(self, in_token,):
        in_proba = self.token2proba(in_token)
        
        self.hid = self.in2hid @ in_proba + self.hid2hid @ self.hid + self.bias_before_activation
        self.hid = np.tanh(self.hid) # TODO WHY???
        out_logproba = self.hid2out @ self.hid + self.out_logproba_bias # why not use input, too? (extra params for no good reason) 
        out_proba = self.softmax(out_logproba) 

        self.hid_history.append(self.hid)

        return out_proba, (in_proba, self.hid, out_logproba,)
    
    def predict_token(self, in_token, temperature):
        out_proba, _ = self.predict_proba(in_token)
        return np.argmax(out_proba)

    # GENERATE
    def generate_encoded(self, seed_phrase=None, max_length=200, temperature=1.0, visualize_activation=False):

        seed_phrase_encoded = self.encode(seed_phrase)

        self.reset_hid()
        seed_output = []
        for seed_token in seed_phrase_encoded:
            seed_output.append(self.predict_token(seed_token, temperature))
        if visualize_activation:
            self.visualize_activation(seed_phrase=seed_phrase)

        
        output = seed_phrase_encoded
        for i in range(max_length):
            output.append(self.predict_token(output[-1], temperature))
        
        return output, seed_output

    def generate(self, seed_phrase=None, max_length=200, temperature=1.0):
        output_encoded, seed_output_encoded = self.generate_encoded(seed_phrase=seed_phrase,
                                                                    max_length=max_length,
                                                                    temperature=temperature)
        return self.decode(output_encoded)
    
    # INSPECT
    def visualize_activation(self, seed_phrase):
        dir4images = self.make_dir4images(seed_phrase)

        for neuron in range(self.hid_dim):
            image = self.VerySmartImage(seed_phrase)
            for t in range(len(seed_phrase)):
               image.draw_character(
                   character=seed_phrase[t],
                   neuron_value=self.hid_history[t][neuron]
               ) 
            image.save(dir4images=dir4images, neuron_name=f'{neuron}.bmp')


    def make_dir4images(self, seed_phrase):
        dir4images = Path(
            f"gigabytes_of_neurons/{self.hid_dim}_lr_{self.learning_rate}_lc_{self.line_count}_ll_{self.line_length}_{seed_phrase}"
        )
        dir4images.mkdir(exist_ok=True)

        # dump params
        params_dict = deepcopy(self.__dict__)
        with open(dir4images / "params.json", "w") as params_file:
            for i in range(3):
                print('###' * 20, file=params_file)

            for item in params_dict:
                if item == 'hid_history':
                    continue
                print(item, file=params_file)
                print(params_dict[item], file=params_file)

        return dir4images

    
    class VerySmartImage:
        def __init__(self, seed_phrase,):
            # initiate canvas
            self.create_image(seed_phrase)

            # current position
            self.x = self.init_x
            self.y = self.init_y

            # colours
            self.red = np.array([255, 0, 0])
            self.blue = np.array([0, 0, 255])
 
        def draw_character(self, character, neuron_value,):
            self.drawer.text((self.x, self.y), self.character_to_draw(character),
                              fill=self.linear_colour_gradient(neuron_value), 
                              font=self.font
                            )

            self.update_position(character)

        def save(self, dir4images, neuron_name,):
            self.image.save(dir4images / f'{neuron_name}')

        def linear_colour_gradient(self, real_from_pm1):
            red_ratio = (real_from_pm1 + 1) / 2
            return tuple((red_ratio * self.red + (1 - red_ratio) * self.blue).astype(int))
        
        def update_position(self, character):
            self.x += self.delta_x
            if character == '\n':
                self.x = self.init_x
                self.y += self.init_y

        def character_to_draw(self, character):
            return character.replace(' ', '_').replace('\n', '_\n')
        
        def create_image(self, seed_phrase):
            self.font_size = 15
            self.init_x = self.font_size + 1
            self.init_y = self.font_size + 1
            self.delta_x = self.font_size + 1
            self.delta_y = self.font_size + 1

            lines = seed_phrase.split('\n')
            self.height = (len(lines) * self.delta_y * 3) // 2 + 2 * self.init_y
            longest_line = max(lines, key=len)
            its_len = len(longest_line)
            self.width = (its_len * self.delta_x * 3) // 2 + 2 * self.init_x

            self.image = Image.new('RGB', (self.width, self.height))
            self.drawer = ImageDraw.Draw(self.image)
            self.font = ImageFont.truetype("UbuntuMono-Regular.ttf", 15, encoding='UTF-8')

    # AUX
    def reset_hid(self):
        self.hid *= 0
    
    def reset_hid_history(self):
        self.hid_history = []

    def encode(self, human_written_text):
        return [self.token_to_idx[x] for x in human_written_text]

    def decode(self, encoded_text):
        return [self.idx_to_token[x] for x in encoded_text]

    def make_chunks(self, text_encoded, line_count, line_length):
        start_index = np.random.randint(0, len(text_encoded) - line_count * line_length)
        chosen_text = text_encoded[start_index:start_index + line_count*line_length]
        #print(''.join(self.decode(chosen_text)))
        data = np.array(chosen_text).reshape((line_count, line_length))

        # TODO add footer?
        start_column = np.zeros((line_count, 1), dtype=int) + self.header_idx

        return np.hstack((start_column, data))

    def token2proba(self, token):
        proba = np.zeros(shape=self.num_tokens)
        proba[token] = 1
        return proba

    def proba2token(self, proba):
        return np.argmax(proba)

    def outer_vector_product(self, v, u):
        return v.reshape(len(v), 1) @ u.reshape(1, len(u))

    def softmax(self, x):
        m = np.max(x)
        ex = np.exp(x - m)
        return ex / sum(ex)
    
    def scales_of_change(self, param_changes):
        scale = 0
        for param, param_delta in param_changes:
            scale += (param_delta*param_delta).sum()
        return np.sqrt(scale)
    
    def hi(self):
        print('hi!!')