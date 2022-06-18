from math import ceil, floor
import random
from gym_multigrid.multigrid import *


class RedBlueDoor(MultiGridEnv):
    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        agents_index=[0, 1],
        zero_sum=False,
        view_size=3,
        see_through_walls=True,
        actions_set=OpenActions
    ):
        self.zero_sum = zero_sum
        self.world = World
        self.side = None

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=512,
            # Set this to True for maximum speed
            see_through_walls=see_through_walls,
            agents=agents,
            agent_view_size=view_size,
            actions_set=actions_set
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        x = random.sample(range(1, height-2), 2)
        y = random.sample(range(1, width-2), 2)

        a1 = self.agents[0]
        a1.pos = (x[0], y[0])
        a1.dir = 1
        self.put_obj(a1, x[0], y[0])
        # self.put_obj(Goal(self.world, a1.index), *corner)

        a2 = self.agents[1]
        a2.pos = (x[1], y[1])
        a2.dir = 1
        self.put_obj(a2, x[1], y[1])

        self.red_door = [0, random.randint(1, height-2)]
        self.put_obj(Door(self.world, "red"), *self.red_door)

        self.blue_door = [width-1, random.randint(1, height-2)]
        self.put_obj(Door(self.world, "blue"), *self.blue_door)

    def step(self, actions):
        # state at previous timestap (t-1)
        red_open = self.grid.get(*self.red_door).is_open

        obs, rewards, done, info = MultiGridEnv.step(self, actions)

        if red_open:
            if self.grid.get(*self.blue_door).is_open and self.grid.get(*self.red_door).is_open:
                success = True
                rewards = [self._reward(i, rewards, 1) for i in range(len(self.agents))]
                done = True
            success = False
        else:
            if self.grid.get(*self.blue_door).is_open:
                done = True
            success = False

        info["success"] = success
        dones = [done] * len(self.agents)
        return obs, rewards, dones, info


class RedBlueDoor6x6(RedBlueDoor):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=6,
                         height=6)


class RedBlueDoor8x8(RedBlueDoor):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=8,
                         height=8)


class RedBlueDoor10x10(RedBlueDoor):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=10,
                         height=10)


class RedBlueDoor12x12(RedBlueDoor):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=12,
                         height=12)
