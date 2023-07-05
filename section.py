from network import Network


class Section:

    def __init__(self, blocks=[]):
        self.section = blocks

    def add_block(self, block: Network):
        self.section.append(block)
