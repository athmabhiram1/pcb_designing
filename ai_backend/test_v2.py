import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
from circuit_schema import CircuitData
from engines.kicad_exporter import KiCadSchematicWriter

circuit_data = {
    'description': 'Simple LED indicator circuit',
    'components': [
        {'ref':'R1','lib':'Device','part':'R','value':'330',
         'footprint':'Resistor_SMD:R_0805_2012Metric',
         'description':'Current limiting resistor',
         'pins':[{'number':'1','name':'1'},{'number':'2','name':'2'}]},
        {'ref':'D1','lib':'Device','part':'LED','value':'Red',
         'footprint':'LED_SMD:LED_0805_2012Metric',
         'description':'Red LED',
         'pins':[{'number':'1','name':'A'},{'number':'2','name':'K'}]},
        {'ref':'C1','lib':'Device','part':'C','value':'100nF',
         'footprint':'Capacitor_SMD:C_0402_1005Metric',
         'description':'Power supply bypass capacitor',
         'pins':[{'number':'1','name':'+'},{'number':'2','name':'-'}]},
    ],
    'connections': [
        {'net':'VCC',       'pins':['R1.1','C1.1']},
        {'net':'LED_ANODE', 'pins':['R1.2','D1.1']},
        {'net':'GND',       'pins':['D1.2','C1.2']},
    ]
}
circuit = CircuitData(**circuit_data)
writer = KiCadSchematicWriter()
sch = writer.export(circuit)

out = os.path.join(os.path.dirname(__file__), 'output', 'test_led_v2.kicad_sch')
with open(out, 'w') as f:
    f.write(sch)

print('Component + power placements:')
for m in re.finditer(r'\(symbol \(lib_id "([^"]+)"\) \(at ([\d.]+) ([\d.]+) ([\d.]+)\)', sch):
    print(f'  {m.group(1):40s} at ({m.group(2)}, {m.group(3)}) rot={m.group(4)}')

print('\nWires:')
for m in re.finditer(r'\(wire \(pts \(xy ([\d.]+) ([\d.]+)\) \(xy ([\d.]+) ([\d.]+)\)', sch):
    print(f'  ({m.group(1)},{m.group(2)}) -> ({m.group(3)},{m.group(4)})')

print('\nLabels:', re.findall(r'\(label "([^"]+)"', sch))
print('\nSaved:', out)
