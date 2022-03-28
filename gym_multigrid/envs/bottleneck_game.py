from math import ceil, floor
import random
from gym_multigrid.multigrid import *


class BottleneckGame(MultiGridEnv):
    """
    Environment in which the agents have pass through a bottleneck.
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        agents_index=[],
        zero_sum=False,
        view_size=5,
        see_through_walls=False,
        fixed_pos=True,
        actions_set=MoveActions,
        goal_zone=2,
        max_steps=512,
    ):
        self.zero_sum = zero_sum
        self.world = World
        self.fixed_pos = fixed_pos
        self.goal_zone = goal_zone

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=max_steps,
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

        num_bottlenecks = int((self.height-3)/2)
        if num_bottlenecks == 1:
            # Add bottleneck wall
            self.grid.horz_wall(self.world, 0, floor(height/2), length=floor(width/2))
            self.grid.horz_wall(self.world, ceil(width/2), floor(height/2), length=floor(width/2))
        else:
            for i in range(num_bottlenecks):
                rand_w = random.choice(range(2, self.width-3))
                wall_h = (i+1)*2
                # First wall
                self.grid.horz_wall(self.world, 0, wall_h, length=rand_w)
                self.grid.horz_wall(self.world, rand_w+1, wall_h, length=self.height-1-rand_w)

        top_corners = [[1, 1], [width-2, 1]]
        bot_corners = [[1, height-2], [width-2, height-2]]
        corners = [top_corners, bot_corners]

        if len(self.agents) < 3:
            if self.fixed_pos:
                rand_corner, rand_a, rand_g = [0, 0, 0]
            else:
                rand_corner, rand_a, rand_g = [
                    random.randint(0, 1),
                    random.randint(0, 1),
                    random.randint(0, 1)
                ]

            for _, a in enumerate(self.agents):

                a_pos = corners[rand_corner][rand_a]
                g_pos = corners[1-rand_corner][rand_g]

                a.pos = (a_pos[0], a_pos[1])
                a.dir = 0
                self.put_obj(a, *a_pos)

                i, j = g_pos
                for k in range(self.goal_zone):
                    if i+k < self.width and self.grid.get(i+k, j) is None:
                        self.put_obj(Goal(self.world, a.index), i+k, j)
                    if i-k >= 0 and self.grid.get(i-k, j) is None:
                        self.put_obj(Goal(self.world, a.index), i-k, j)

                rand_corner = 1 - rand_corner
                temp = rand_a
                rand_a = 1 - rand_g
                rand_g = 1 - temp
        else:
            corners = [*top_corners, *bot_corners]
            rand_corners = random.sample(corners, len(self.agents))
            for id, a in enumerate(self.agents):
                a_pos = rand_corners[id]
                g_pos = [rand_corners[id][0], height-1-rand_corners[id][1]]

                a.pos = (a_pos[0], a_pos[1])
                a.dir = 0
                if isinstance(self.grid.get(*a_pos), Goal):
                    g = self.grid.get(*a_pos)
                    a.standing_on = g
                self.put_obj(a, *a_pos)

                i, j = g_pos
                for k in range(self.goal_zone):
                    if i+k < self.width:
                        fwd_cell = self.grid.get(i+k, j)
                        if fwd_cell is None:
                            self.put_obj(Goal(self.world, a.index), i+k, j)
                        elif isinstance(fwd_cell, Goal):
                            fwd_cell.indices.append(a.index)
                        elif isinstance(fwd_cell, Agent):
                            if fwd_cell.standing_on is None:
                                fwd_cell.standing_on = Goal(self.world, a.index)
                            else:
                                fwd_cell.standing_on.indices.append(id)
                    if i-k >= 0: 
                        fwd_cell = self.grid.get(i-k, j)
                        if fwd_cell is None:
                            self.put_obj(Goal(self.world, a.index), i-k, j)
                        elif isinstance(fwd_cell, Goal):
                            fwd_cell.indices.append(a.index)
                        elif isinstance(fwd_cell, Agent):
                            if fwd_cell.standing_on is None:
                                fwd_cell.standing_on = Goal(self.world, a.index)
                            else:
                                fwd_cell.standing_on.indices.append(id)

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        success = all([_reached_goal(a) for a in self.agents])
        done = done or success
        if success:
            rewards = [self._reward(i, rewards, 1) for i in range(len(self.agents))]
        return obs, rewards, done, info


def _reached_goal(agent: Agent) -> bool:
    """Returns true if the agent is standing on the correct goal."""
    if agent.standing_on:
        return agent.index in agent.standing_on.indices
    else:
        return False


class BottleneckGame1A5x5(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False,
                         fixed_pos=False)


class BottleneckGame1A5x5F(BottleneckGame):
    def __init__(self):
        super().__init__(size=5,
                         agents_index=[0],
                         zero_sum=False,
                         fixed_pos=True)


class BottleneckGame2A7x5F(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=5,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions)


class BottleneckGame2A7x5(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=5,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions)


class BottleneckGame3A7x5Z(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1, 2],
                         zero_sum=False,
                         width=7,
                         height=5,
                         see_through_walls=False,
                         fixed_pos=False,
                         goal_zone=10,
                         actions_set=MoveActions, 
                         max_steps=128)


class BottleneckGame2A7x7F(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions)


class BottleneckGame2A7x7(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions)


class BottleneckGame2A7x7FZ(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions,
                         goal_zone=10)


class BottleneckGame2A7x7Z(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=7,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions,
                         goal_zone=10)


class BottleneckGame2A15x15FZ(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=15,
                         height=15,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions,
                         goal_zone=15)


class BottleneckGame2A15x15Z(BottleneckGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=15,
                         height=15,
                         see_through_walls=False,
                         fixed_pos=False,
                         actions_set=MoveActions,
                         goal_zone=15)
