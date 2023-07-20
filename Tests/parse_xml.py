import xml.etree.ElementTree as ElementTree

tripinfos = ElementTree.parse('./Data0-0/tripinfo.xml').getroot().findall('tripinfo')
trips = 0

CO_abs, CO2_abs, HC_abs, PMx_abs, NOx_abs, fuel_abs, elec_abs = 0,0,0,0,0,0,0


for tripinfo in tripinfos:
    emission = tripinfo.findall('emissions')[0]
    CO_abs += float(emission.get('CO_abs'))
    CO2_abs += float(emission.get('CO2_abs'))
    HC_abs += float(emission.get('HC_abs'))
    PMx_abs += float(emission.get('PMx_abs'))
    NOx_abs += float(emission.get('NOx_abs'))
    fuel_abs += float(emission.get('fuel_abs'))
    elec_abs += float(emission.get('electricity_abs'))
    trips+=1

print(tripinfos)    
print(trips)
print(CO_abs)