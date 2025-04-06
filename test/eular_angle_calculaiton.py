from utils.math_tools import calculate_delta_position

from_rxyz = [0,0,0,2.945261329843815, -0.676870785350858, -1.8704226650156563]
to_rxyz = [0,0,0,3.1415926,0.0,1.5707963]

print(calculate_delta_position(from_rxyz,to_rxyz))