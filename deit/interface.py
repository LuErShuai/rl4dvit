import torch

Class InterfaceRL4Deit():
    def __init__(self, condition, agent):
        self.condition = condition
        self.agent = Agent()

    def send_step_signal_to_rl():

    def send_state_to_rl():
        # if store state is needed

        # if train process to be excuted



Class InterfaceDeit4RL():
    def __init__(self, condition, deit):
        self.condition = condition

    def wait_step_signal_from_deit():

    def receive_state_from_deit():
        with self.condition:

        


  
        # with self.condition:
        #     condition.wait()
        #     # print("RL   : Deit, d u need a mask?")
        #     print("Deit : yes!")
        #     condition.notify()

        #     conditin.wait()
        #     # print("RL   : pass me the state please.")
        #     print("Deit : here u go.")
        #     self.send_state_to_rl(x)
        #     condition.notify()

        #     condition.wait()
        #     # print("RL   : got it! here is the mask that you want.")
        #     mask = self.get_mask_from_rl()
        #     # mask = self.get_mask_from_rl_agent(x)
        #     condition.notify()

        # # get obs from deit
        # with self.condition:
        #     print("RL : Deit, d u need a mask?")
        #     condition.notify()
        #     condition.wait()
        #     # print("Deit : yes!")

        #     print("RL   : pass me the state please.")
        #     conditon.notify()
        #     condition.wait()
        #     # print("Deit : here u go.")

        #     state = self.get_state_from_deit()
        #     action = choose_action(state)
        #     
        #     set_mask_for_deit(action)
        #     print("RL   : got it! here is the mask that you want.")
        #     condition.notify()
        #     condition.wait()

