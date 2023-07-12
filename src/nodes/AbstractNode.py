import src.user_interaction as usr_int
import src.file_handler as file_handler
import cv2

class AbstractNode:
    """
    Every child of abstractnode has to define process and initialize_node.
    If you have output to save define save_output.
    Its suggested to replace reinitialize_node
    Its only necessary to replace plot_output if you allow users to retry
    """
    NODE_SUCCESS_CODE = 1000
    NODE_FAILURE_CODE = 1001
    def __init__(self,
                 output_name="",
                 requirements=[],
                 user_can_retry=False,
                 node_title="Uninitialized Node Title"):
        self.requirement_exists = {}
        self.output_name = output_name
        self.requirements = requirements
        self.user_can_retry = user_can_retry
        self.node_title = node_title
        self.requirements_met = True
        self.node_status_code = AbstractNode.NODE_FAILURE_CODE
        self.output_pack = None
        

    def get_output_name(self):
        return self.output_name

    def set_node_as_successful(self):
        self.node_status_code = AbstractNode.NODE_SUCCESS_CODE

    def set_node_as_failed(self):
        self.node_status_code = AbstractNode.NODE_FAILURE_CODE

    def process(self):
        print("This is the default process method that does nothing")
        return None

    def initialize_node(self):
        print("This is the default initialization method that does nothing")

    def reinitialize_node(self):
        self.initialize_node()

    def plot_output(self):
        print("This is the default plot method that does nothing")

    def ask_user_if_they_have_substitute_for_requirement(self, requirement):
        prompt = "The requirement {requirement} has not been met by a"
        prompt += "step earlier in the pipeline. Do you have a replacement?"
        

    def check_requirements(self):
        from src.fishnet import FishNet
        for requirement in self.requirements:
            if requirement not in FishNet.pipeline_output.keys():
                user_response_id = usr_int.ask_if_user_has_replacement_for_requirement(requirement)
                if user_response_id == usr_int.positive_response_id:
                    loaded_img = file_handler.load_img_file()
                    FishNet.pipeline_output[requirement] = loaded_img
                elif user_response_id == usr_int.negative_response_id:
                    self.requirements_met = False

    def node_intro_msg(self):
        prompt = f"\n---- Commencing {self.node_title} ----\n"
        print(prompt)

    def give_and_save_node_data(self):
        if self.node_status_code == AbstractNode.NODE_SUCCESS_CODE:
            self.save_output()
            self.give_fishnet_output()

    def save_output(self):
        pass

    def give_fishnet_output(self):
        from src.fishnet import FishNet
        FishNet.store_output(self.output_pack, self.output_name)

    def save_img(self, img, img_name, cmap="gray"):
        from src.fishnet import FishNet
        folder_name = FishNet.save_folder
        img_file_path = folder_name + img_name
        cv2.imwrite(img_file_path, img)
        # fig, ax = plt.subplots(figsize=(16, 16))
        # plt.axis("off")
        # ax.imshow(img, cmap=cmap)
        # plt.savefig(img_file_path, bbox_inches="tight")
        # plt.close(fig)

    def run(self):
        self.node_intro_msg()
        self.check_requirements()
        if self.requirements_met is False:
            return None

        self.initialize_node()
        if self.user_can_retry:
            usr_feedback = usr_int.retry_response_id
            first_pass = True
            while usr_feedback == usr_int.retry_response_id:
                if not first_pass:
                    self.reinitialize_node()

                node_output = self.process()
                self.plot_output()
                usr_feedback = usr_int.get_user_feedback_for_node_output()
                # Close output maybe?

                if usr_feedback == usr_int.satisfied_response_id:
                    self.give_and_save_node_data()
                    return self.node_status_code
                    # return self.node_status_code # eventually this is what we return
                elif usr_feedback == usr_int.quit_response_id:
                    return self.node_status_code
                if first_pass:
                    first_pass = False
        else:
            node_output = self.process()
            self.give_and_save_node_data()
            return self.node_status_code

