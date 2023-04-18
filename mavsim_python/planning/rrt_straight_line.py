# rrt straight line path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - Brady Moon
#         4/11/2019 - RWB
#         3/31/2020 - RWB
import numpy as np
from message_types.msg_waypoints import MsgWaypoints
from viewers.planner_viewer import PlannerViewer

class RRTStraightLine:
    def __init__(self, app, show_planner=True):
        self.segment_length = 300 # standard length of path segments
        self.show_planner = show_planner
        if show_planner:
            self.planner_viewer = PlannerViewer(app)

    def update(self, start_pose, end_pose, Va, world_map, radius):
        tree = MsgWaypoints()
        # tree.type = 'straight_line'
        tree.type = 'fillet'

        ###### TODO ######
        # add the start pose to the tree
        tree.add(start_pose, airspeed=Va, parent=-1)
        
        # check to see if start_pose connects directly to end_pose
        if not collision(start_pose, end_pose, world_map):
            tree.add(end_pose, airspeed=Va, parent=0)
            return tree
        
        connected_to_end = False
        while not connected_to_end:
        # for i in range(3):
            self.extend_tree(tree, end_pose, Va, world_map)
            if tree.connect_to_goal[-1]:
                connected_to_end = True
        # print(tree.num_waypoints)
        # exit()
            
        
        # find path with minimum cost to end_node
        # waypoints_not_smooth = find_minimum_path()
        waypoints = smooth_path(tree, world_map)
        waypoints.type = tree.type
        # waypoints = MsgWaypoints()
        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map):
        # extend tree by randomly selecting pose and extending tree toward that pose
        
        # Get a random pose within the world boundaries
        rand_pose = random_pose(world_map, -100).flatten()
        # Find the closest node in the tree to the random pose
        for i in range(tree.num_waypoints):
            dist = distance(tree.ned[0:2, i], rand_pose[0:2])
            if i == 0:
                min_dist = dist
                min_idx = i
            elif dist < min_dist:
                min_dist = dist
                min_idx = i
        # Extend the tree toward the random pose
        unit_vector_to_pose = (rand_pose - tree.ned[:, min_idx]) / min_dist
        new_pose = tree.ned[:, min_idx] + unit_vector_to_pose * self.segment_length
        if not collision(tree.ned[:, min_idx], new_pose, world_map):
            new_pose = new_pose.reshape(3, 1)
            if distance(new_pose, end_pose) < self.segment_length and not collision(new_pose, end_pose, world_map):
                tree.add(new_pose, airspeed=Va, parent=min_idx, connect_to_goal=1)
                tree.add(end_pose, airspeed=Va, parent=tree.num_waypoints-1, connect_to_goal=1)
            else:
                tree.add(new_pose, airspeed=Va, parent=min_idx)
        
        ###### TODO ######
        flag = None
        return flag
        
    def process_app(self):
        self.planner_viewer.process_app()

def smooth_path(waypoints, world_map):

    ##### TODO #####
    # smooth the waypoint path
    # smooth = [0]  # add the first waypoint
    smooth = [waypoints.num_waypoints-1]  # add the last waypoint
    while smooth[-1] != 0:
        parent_idx = int(waypoints.parent[smooth[-1]])
        while True:
            parent_of_parent_idx = int(waypoints.parent[parent_idx])
            if collision(waypoints.ned[:, parent_of_parent_idx], waypoints.ned[:, smooth[-1]], world_map):
                break
            parent_idx = parent_of_parent_idx
            if parent_idx == 0:
                break
        smooth.append(parent_idx)
    smooth = smooth[::-1] # Flip the list
    
    # construct smooth waypoint path
    smooth_waypoints = MsgWaypoints()
    for idx in smooth:
        waypoint = waypoints.ned[:, idx].reshape(3, 1)
        airspeed = waypoints.airspeed[idx]
        smooth_waypoints.add(waypoint, airspeed=airspeed)

    return smooth_waypoints


def find_minimum_path(tree, end_pose):
    # find the lowest cost path to the end node

    ##### TODO #####
    # find nodes that connect to end_node
    connecting_nodes = []
    
    # find minimum cost last node
    idx = 0

    # construct lowest cost path order
    path = []  # last node that connects to end node
    
    # construct waypoint path
    waypoints = MsgWaypoints()
    return waypoints


def random_pose(world_map, pd):
    # generate a random pose

    ##### TODO #####
    pn = np.random.uniform(0, world_map.city_width)
    pe = np.random.uniform(0, world_map.city_width)
    pose = np.array([[pn], [pe], [pd]])
    return pose


def distance(start_pose, end_pose):
    # compute distance between start and end pose

    ##### TODO #####
    d = np.linalg.norm(end_pose - start_pose)
    return d


def collision(start_pose, end_pose, world_map):
    # check to see of path from start_pose to end_pose colliding with map
    def point_in_building(point):
        north = np.abs(point.item(0) - world_map.building_north) < world_map.building_width / 2
        east = np.abs(point.item(1) - world_map.building_east) < world_map.building_width / 2
        if np.any(north) and np.any(east):
            nidx = list(north[0,:]).index(True)
            eidx = list(east[0,:]).index(True)
            if point.item(2) < world_map.building_height[nidx, eidx]:
                return True
        return False
        
    # Report a collision if the end pose is outside the city    
    if end_pose.item(0) < 0 or end_pose.item(0) > world_map.city_width or \
            end_pose.item(1) < 0 or end_pose.item(1) > world_map.city_width:
        return True
    
    # Report a collision if the end pose is inside a building
    if point_in_building(end_pose):
        return True
    
    # Otherwise, step along the path and check for collisions
    step_dist = 2
    num_points = int(distance(start_pose, end_pose) / step_dist)
    points_to_check = points_along_path(start_pose, end_pose, num_points)
    for i in range(num_points):
        point = points_to_check[:, i]
        if point_in_building(point):
            return True

    return False

def height_above_ground(world_map, point):
    # find the altitude of point above ground level
    
    ##### TODO #####
    h_agl = 0
    return h_agl

def points_along_path(start_pose, end_pose, N):
    # returns points along path separated by Del
    sx, sy, sz = start_pose.item(0), start_pose.item(1), start_pose.item(2)
    ex, ey, ez = end_pose.item(0), end_pose.item(1), end_pose.item(2)
    xs = np.linspace(sx, ex, N)
    ys = np.linspace(sy, ey, N)
    zs = np.linspace(sz, ez, N)
    points = np.vstack((xs, ys, zs))
    return points


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col