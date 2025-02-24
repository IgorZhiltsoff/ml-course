import string
import os
import sys

from random import sample

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

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
        self.grad_length_history = []
            # for visualizing activation
        self.hid_history = []
        self.proba_history = []
    
    # FIT 
    def fit(self, human_written_text, 
            line_count=256, line_length=100,
            iterations=1,
            learning_rate=0.3, l2_reg=0.0, clip=None,):

        iterations_ellapsed = 0
        text_encoded = self.encode(human_written_text)
        for it in range(iterations):
            inputs = self.make_chunks(text_encoded, line_count=line_count, line_length=line_length)

            targets = [x[1:] for x in inputs]
            inputs = [x[:-1] for x in inputs]

            iterations_ellapsed = self.fit_with_chunk(inputs, targets, 
                                        learning_rate=learning_rate, l2_reg=l2_reg, clip=clip, iterations_ellapsed=iterations_ellapsed)

        # history
        self.learning_rate = learning_rate
        self.line_count = line_count
        self.line_length = line_length
        self.l2_reg = l2_reg
        self.clip = clip
        self.iterations = iterations


    def fit_with_chunk(self, 
            inputs,
            targets,
            learning_rate, l2_reg, clip,
            iterations_ellapsed
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

            # forward pass
            for t in range(n):
                out_proba, (in_proba, out_logproba,) = self.predict_proba(input_line[t])

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

            # don't worry, those are refs :)
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

                dloss_d_out_logproba = np.copy(out_proba_at[t])
                # everything except for targets and hiddens is indexed from 1
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
                dloss_d_hid_at_t_plus_1 = self.hid2hid.T @ dloss_d_preactivated_hid_at_t

            # update params
            for param, param_delta in param_changes:
                if clip:
                    param_delta = np.clip(param_delta, -clip, clip) 
                param += -learning_rate * param_delta

            self.loss_history.append(loss)
            self.grad_length_history.append(self.grad_length(param_changes))
            if it % 100 == 0:
                print(f'Loss on iteration {iterations_ellapsed} is {loss}')
                print(f'Length of gradient is {self.grad_length_history[-1]}')
                print()
            iterations_ellapsed += 1
        return iterations_ellapsed

    def show_loss_on_train(self,):
        plt.plot(self.loss_history)

    def show_grad_length_on_train(self,):
        plt.plot(self.grad_length_history)


    # PREDICT
    def predict_proba(self, in_token,):
        in_proba = self.token2proba(in_token)
        
        self.hid = self.in2hid @ in_proba + self.hid2hid @ self.hid + self.bias_before_activation
        self.hid = np.tanh(self.hid) # TODO WHY???
        out_logproba = self.hid2out @ self.hid + self.out_logproba_bias # why not use input, too? (extra params for no good reason) 
        out_proba = self.softmax(out_logproba) 

        # history for visualization
        self.hid_history.append(self.hid)
        self.proba_history.append(out_proba)

        return out_proba, (in_proba, out_logproba,)
    
    def predict_token(self, in_token, temperature):
        out_proba, _ = self.predict_proba(in_token)
        return np.argmax(out_proba)

    # GENERATE
    # FIXME visualize param
    def generate_encoded(self, seed_phrase=None, max_length=200, temperature=1.0, visualize_activation=False):

        seed_phrase_encoded = self.encode(seed_phrase)

        self.reset_hid()
        self.reset_history_for_visualization()
        seed_output = []
        for seed_token in seed_phrase_encoded:
            seed_output.append(self.predict_token(seed_token, temperature))
        if visualize_activation:
            self.visualize_activation(seed_phrase=seed_phrase)

        
        output = seed_phrase_encoded
        for i in range(max_length):
            output.append(self.predict_token(output[-1], temperature))

        #print(f'hi{len(output)}')
        
        return output, seed_output

    def generate(self, seed_phrase=None, max_length=200, temperature=1.0):
        return self.generate_with_seed_output(seed_phrase=seed_phrase,
                                              max_length=max_length,
                                              temperature=temperature,
                                              )[0]
    
    def generate_with_seed_output(self, seed_phrase, max_length, temperature):
        output_encoded, seed_output_encoded = self.generate_encoded(seed_phrase=seed_phrase,
                                                                    max_length=max_length,
                                                                    temperature=temperature)
        return self.decode(output_encoded), self.decode(seed_output_encoded)
    
    # INSPECT
    def visualize_activation(self, seed_phrase, use_output=True, max_length=200,
                             #visualize_proba=True,
                             verbose=False
                             ):
        output, seed_output = self.generate_with_seed_output(
            seed_phrase=seed_phrase,
            max_length=max_length,
            temperature=None # TODO choose appropriate value
        )
        if verbose:
            print('Generated')

        dir4images = self.make_dir4images(seed_phrase)

        text = output 

        print(len(text))
        for neuron in range(self.hid_dim):
            if verbose:
                print(f'neuron {neuron}')

            self.visualize_single_neuron(text=text, values=[
                self.hid_history[t][neuron] 
                for t in range(len(text))], 
                                         name=neuron, dir4images=dir4images)

        proba_of_drawn_character = np.array([
            self.proba_history[t][self.token_to_idx[text[t]]]
            for t in range(len(text))
        ])
        # make it in [-1, 1]
        proba_of_drawn_character *= 2
        proba_of_drawn_character -= 1

        self.visualize_single_neuron(text=text, values=proba_of_drawn_character,
                name='proba', dir4images=dir4images)
        if verbose:
            print('Done')

    def visualize_single_neuron(self, text, values, name, dir4images):
        self.image = self.VerySmartImage(text)
        for t in range(len(text)):
            self.image.draw_character(
                character=text[t],
                neuron_value=values[t]
            ) 
        self.image.save(dir4images=dir4images, name=name)


    def make_dir4images(self, seed_phrase):
        dir4images = Path(
            f"gigabytes_of_neurons/{self.hid_dim}_lr_{self.learning_rate}_lc_{self.line_count}_ll_{self.line_length}_it_{self.iterations}_{seed_phrase}".replace(
                                  '\n', '___'                        
            )
        )
        dir4images.mkdir(exist_ok=True)

        # dump params
        forbidden_keys = ['hid_history', 'proba_history', 'image']
        params_dict = deepcopy({key : self.__dict__[key] for key in self.__dict__ if key not in forbidden_keys})
        with open(dir4images / "params.txt", "w") as params_file:
            for i in range(3):
                print('###' * 20, file=params_file)

            for item in params_dict:
                print(item, file=params_file)
                print(params_dict[item], file=params_file)

        return dir4images

    
    class VerySmartImage:
        def __init__(self, text,):
            # initiate canvas
            self.create_image(text)

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

        def save(self, dir4images, name,):
            self.image.save(dir4images / f'{name}.bmp')

        def linear_colour_gradient(self, real_from_pm1):
            red_ratio = (real_from_pm1 + 1) / 2
            return tuple((red_ratio * self.red + (1 - red_ratio) * self.blue).astype(int))
        
        def update_position(self, character):
            self.x += self.delta_x
            if self.x > self.width - 2 * self.font_size:
                self.x = self.init_x
                self.y += self.delta_y

        def character_to_draw(self, character):
            return character.replace(' ', '_').replace('\n', '#')
        
        def create_image(self, text):
            self.font_size = 15 # 25 # TODO calculate optimal
            self.init_x = self.font_size + 1
            self.init_y = self.font_size + 1
            self.delta_x = self.font_size + 1
            self.delta_y = self.font_size + 1

            lines = text.split('\n')
            real_height = len(lines) * (self.font_size + self.delta_y) + 2 * self.init_y
            longest_line = max(lines, key=len)
            its_len = len(longest_line)
            real_width = its_len * (self.font_size + self.delta_x) + 2 * self.init_x

            mult_x = (self.font_size + self.delta_x) 
            shift_x = 2 * self.init_x
            mult_y = (self.font_size + self.delta_y) 
            shift_y = 2 * self.init_y

            unscaled = int(np.sqrt(len(text))) + 1 # TODO what is wrong with bottom?
            self.width = mult_x * unscaled + shift_x
            self.height = mult_y * unscaled + shift_y


            self.image = Image.new('RGB', (self.width, self.height))
            self.drawer = ImageDraw.Draw(self.image)
            self.font = ImageFont.truetype("UbuntuMono-Regular.ttf", self.font_size, encoding='UTF-8')

    # AUX
    def reset_hid(self):
        self.hid *= 0
    
    def reset_history_for_visualization(self):
        self.hid_history = []
        self.proba_history = []

    def encode(self, human_written_text):
        return [self.token_to_idx[x] for x in human_written_text]

    def decode(self, encoded_text):
        return ''.join([self.idx_to_token[x] for x in encoded_text])

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
        return ex / np.sum(ex)
    
    def grad_length(self, param_changes):
        scale = 0
        for param, param_delta in param_changes:
            scale += (param_delta*param_delta).sum()
        return np.sqrt(scale)
    
    def hi(self):
        print('hi!!')