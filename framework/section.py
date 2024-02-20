from typing import List

from framework.block import Block


class Section:
    def __init__(self, blocks: List[Block]):
        self.blocks = blocks
