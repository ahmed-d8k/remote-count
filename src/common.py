import time

## For now these are just wrappers and don't do much. Might delete in the future.
## All functions inherit off of the base Node class below.
## My plan was to save some data on node execution like execution time, success, etc.



class Node:
    def __init__(self):
        pass

    # def process(self, image):
    #     start_time = time.time()
    #     # Call the implementation-specific processing code
    #     self._process(image)
    #     end_time = time.time()
    #     print(f"Processing took {end_time - start_time} seconds")


    def process(self, image):
        # This method should be implemented by the subclasses
        pass


class Pipeline:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)
        
    def process(self, image):
        for node in self.nodes:
            image = node.process(image)
        return image

class TempPipeline:
    def __init__(self, nodes):
        self.node_idx = 0
        self.nodes = nodes
        self.end_node = "ENDNODE"
        self.nodes.append(self.end_node)

    def advance(self):
        # import gc
        node = self.nodes[self.node_idx]
        if node is not self.end_node:
            # del self.nodes[self.node_idx]
            # gc.collect()
            self.node_idx += 1
            return True
        else:
            return False

    def is_not_finished(self):
        node = self.nodes[self.node_idx]
        if node == self.end_node:
            return False
        else:
            return True

    def add_node(self, node):
        self.nodes.append(node)

    def run_node(self):
        node = self.nodes[self.node_idx]
        if node is not self.end_node:
            node_exit_code = node.run()
            node = self.nodes[self.node_idx] #This may be a useless line
            return node_exit_code
