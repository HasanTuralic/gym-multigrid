from math import ceil, floor
import random
from gym_multigrid.multigrid import *


class CommGame(MultiGridEnv):
    """
    Environment in which the agents have pass through a bottleneck.
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        agents_index=[0, 1],
        zero_sum=False,
        view_size=5,
        see_through_walls=False,
        fixed_pos=True,
        actions_set=SmallActions,
        goal_zone=1
    ):
        self.zero_sum = zero_sum
        self.world = World
        self.fixed_pos = fixed_pos
        self.goal_zone = goal_zone
        self.side = None

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=32,
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

        # Add separation wall
        self.grid.horz_wall(self.world, 0, floor(height/2), length=width)

        # agent_pos = [[ceil(width/2), 1], [ceil(width/2), height-2]]
        # bot_corners = [[1, height-2], [width-2, height-2]]
        side = random.randint(0, 1)
        top_corners = [[1, 1], [width-2, 1]]

        a1 = self.agents[0]
        a1.pos = (floor(width/2), 1)
        a1.dir = 1
        self.put_obj(a1, floor(width/2), 1)
        # First agent has one goal
        corner = top_corners[side]
        self.put_obj(Goal(self.world, a1.index), *corner)

        a2 = self.agents[1]
        a2.pos = (floor(width/2), height-2)
        a2.dir = 1
        self.put_obj(a2, floor(width/2), height-2)
        # Second agent has two goals
        self.put_obj(Goal(self.world, a2.index), 1, height-2)
        self.put_obj(Goal(self.world, a2.index), width-2, height-2)

        # Save side to pass as info
        self.side = side

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        temp_done, success = self.check_success(*self.agents)
        done = done or temp_done
        if success:
            rewards = [self._reward(i, rewards, 1) for i in range(len(self.agents))]
        info["side"] = self.side
        info["success"] = success
        return obs, rewards, done, info

    def check_success(self, agent_a: Agent, agent_b: Agent):
        """
        Returns two booleans. 
        The first one signalizes if the episode is done. 
        The second one shows if the episode was finished successfully.
        """
        correct_goal = 1 if self.side == 0 else self.grid.width-2
        if agent_b.standing_on:
            if agent_b.pos[0] == correct_goal:
                # Both agents are on the correct goal
                if agent_a.standing_on and agent_a.pos[0] == agent_b.pos[0]:
                    return True, True
            else:
                return True, False
        return False, False


class CommGame2A7x5(CommGame):
    def __init__(self):
        super().__init__(size=None,
                         agents_index=[0, 1],
                         zero_sum=False,
                         width=7,
                         height=5,
                         see_through_walls=False,
                         fixed_pos=True,
                         actions_set=MoveActions)
