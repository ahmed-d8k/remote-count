from nd2reader import ND2Reader
import cv2 as cv
import numpy as np
import src.user_interaction as usr_int
import sys
import os
import shutil
from src.common import TempPipeline
from src.nodes.SamCellDotCounter import SamCellDotCounter
from src.wrappers.local_sam import LocalSam



class SampleNode():
   def __init__(self):
      self.output_name = "SampleName"

   def get_output_name(self):
      return self.output_name

   def process(self):
      return 0

class ImproperAmountOfNpyFilesFound(Exception):
   def __init__(self):
      msg = "This program expects only 2 npy files to be present in the folder"
      super().__init__(msg)

class MissingCytoOrNucError(Exception):
   def __init__(self):
      msg = "This program expects the two files: cyto_id_mask.npy and "
      msg += "nuc_id_mask.npy. One or both of these files were not found."
      super().__init__(msg)

class FishNet():
   sam_model = LocalSam()
   raw_imgs = []
   channel_meta = {}
   z_meta = 0
   pipeline_output = {}
   save_folder = "output/"
   def __init__(self):
      if not os.path.exists(FishNet.save_folder):
          os.makedirs(FishNet.save_folder)
      elif os.path.exists(FishNet.save_folder):
          shutil.rmtree(FishNet.save_folder)
          os.makedirs(FishNet.save_folder)
      self.placeholder = 0
      self.version = 0.01
      self.valid_file_types = ["nd2"]
      self.img_file = ""
      self.nodes = [SamCellDotCounter()]
      self.pipeline = TempPipeline(self.nodes)
      del self.nodes
      self.valid_responses = ["yes", "y", "no", "n"]
      self.negative_responses = ["no", "n"]
      self.positive_responses = ["yes", "y"]

      self.invalid_response_id = 0
      self.positive_response_id = 1
      self.negative_response_id = 2

      self.welcome_message = f"Welcome to Remote SAM Dot Counter v{self.version}!"
      self.goodbye_message = f"Thank you for using Remote SAM Dot Counter! Goodbye."


   def run(self):
      self.welcome()
      self.prompt_user_for_file()
      self.extract_img_info()
      self.extract_id_masks()
      self.run_pipeline()
      self.goodbye()

   def extract_id_masks(self):
      folder_path = input("Enter path of mask_id_folder: ")
      file_list = os.listdir(folder_path)
      valid_ext = ".npy"
      valid_file_list = []
      mask_pack = {}
      nuc_mask_file_name = "nuc_id_mask.npy"
      cyto_mask_file_name = "cyto_id_mask.npy"
      nuc_mask_path = None
      cyto_mask_path = None
      
      for file_name in file_list:
         if valid_ext in file_name:
            valid_file_list.append(file_name)
      if len(valid_file_list) != 2:
         raise ImproperAmountOfNpyFilesFound
      
      has_cyto = False
      has_nuc = False
      if nuc_mask_file_name in valid_file_list:
         nuc_mask_path = os.path.join(folder_path, nuc_mask_file_name)
         has_nuc = True
      if cyto_mask_file_name in valid_file_list:
         cyto_mask_path = os.path.join(folder_path, cyto_mask_file_name)
         has_cyto = True
          
      if has_nuc and has_cyto:
         mask_pack["cytoplasm"] = np.load(cyto_mask_path)
         mask_pack["nucleus"] = np.load(nuc_mask_path)
         FishNet.pipeline_output["ManualCellMaskPack"] = mask_pack
      else:
         raise MissingCytoOrNucError

   def welcome(self):
      print(self.welcome_message)

   def user_exit(self):
      print("An exit input was recieved, program will now terminate.")
      self.goodbye()
      sys.exit()
   
   # Assuming that a node outputs exactly what the next node wants
   def run_pipeline(self):
      from src.nodes.AbstractNode import AbstractNode
      pipeline_advanced = True
      while(self.pipeline.is_not_finished()):
         node_status_code = self.pipeline.run_node()
         if node_status_code == AbstractNode.NODE_FAILURE_CODE:
             self.user_exit() 
         # Reminder this will break behavior of all previous nodes
         # if node_output is None: # Likely have to change this
         #    self.user_exit()
         # self.store_output(node_output, out_name)
         self.pipeline.advance()


   def process_user_input(self, user_input):
      user_input = user_input.lower()
      if user_input in self.valid_responses:
         if user_input in self.positive_responses:
            return self.positive_response_id
         elif user_input in self.negative_responses:
            return self.negative_response_id
      else:
         return self.invalid_response_id


   def ask_user_to_try_again_or_quit(self):
      prompt = "Would you like to try this step again?\n"
      prompt += "If you say no the program will assume you are done and exit. "
      user_input = input(prompt)
      response_id = self.process_user_input(user_input)
      if response_id == self.positive_response_id:
         return True
      elif response_id == self.negative_response_id:
         return False

   def check_if_user_satisified(self):
      # Display Img
      prompt = "Are you satisfied with the displayed image"
      prompt += " for this step? "
      response_id = self.invalid_response_id
      while(response_id == self.invalid_response_id):
         user_input = input(prompt)
         response_id = self.process_user_input(user_input)
         if response_id == self.positive_response_id:
            return True
         elif response_id == self.negative_response_id:
            return False
         elif response_id == self.invalid_response_id:
            print("Invalid response try again.")
            print("We expect either yes or no.")

   def goodbye(self):
      print(self.goodbye_message)

   def store_output(self, output, out_name):
      FishNet.pipeline_output[out_name] = output

   def store_output(output, out_name):
      FishNet.pipeline_output[out_name] = output

   def prompt_user_for_file(self):
      self.img_file = input("Input nd2 file path: ")

   def convert_list_to_dict(self, arg_list):
      final_dict = {}
      for i in range(len(arg_list)):
         k = arg_list[i]
         final_dict[k] = i
      return final_dict

   def extract_img_info(self):
      with ND2Reader(self.img_file) as images: 
         channel_info = images.metadata["channels"]
         z_len = len(images.metadata["z_levels"])
         z_info = [str(x) for x in range(1, z_len+1)]
         c_len = len(channel_info)
         
         FishNet.z_meta = self.convert_list_to_dict(z_info)
         FishNet.channel_meta = self.convert_list_to_dict(channel_info)
         images.iter_axes = 'zc'
         # z_stack = int(input("Specify how many z slices: "))
         # c_stack = int(input("Specify how many experiment channels: "))
         FishNet.raw_imgs = []
         for z in range(z_len):
            FishNet.raw_imgs.append([])
            for c in range(c_len):
               FishNet.raw_imgs[z].append(images[z*c_len + c])
      FishNet.raw_imgs = np.asarray(FishNet.raw_imgs)

if __name__ == '__main__':
   f = FishNet()
   f.run()
