import krpc 
from time import sleep
from json import dump


sleep(2)
database = []
karmen_line = False
connection = krpc.connect(name="protonk") #  104500.5195214228
vessel = connection.space_center.active_vessel
start_time = connection.space_center.ut
cur_time = connection.add_stream(getattr, connection.space_center, 'ut')
altitude = connection.add_stream(getattr, vessel.flight(), "mean_altitude")
vessel.control.sas = True
vessel.control.throttle = 1.0
print(vessel.dry_mass)
while (connection):
    relative_time = cur_time() - start_time
    print(relative_time, altitude(), vessel.mass, vessel.mass - vessel.dry_mass)
    sleep(0.25)
    if (altitude() >= 41500):
        vessel.control.throttle = 0
    if (altitude() >= 100000 and vessel.flight().vertical_speed <= 10):
        karmen_line = True
        vessel.control.activate_next_stage()
    database.append([relative_time, altitude()])
    if (karmen_line and altitude() < 300):
        connection = False

with open('data.json', 'w') as f:
    dump(database, f)
