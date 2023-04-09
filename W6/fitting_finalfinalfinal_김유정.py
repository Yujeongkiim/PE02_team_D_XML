import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np


# XML 파일에서 Voltage 값과 Current 값을 가져오기
tree = ET.parse('HY202103_D07_(0,0)_LION1_DCM_LMZC.xml')
root = tree.getroot()

#IV Curve
for Voltage in root.iter('Voltage'):
    v = Voltage.text
    vl = list(map(float, v.split(',')))
    print(vl)

for Current in root.iter('Current'):
    i = Current.text
    il = list(map(abs, map(float, i.split(','))))
    print(il)



plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.plot(vl, il, 'bo-', label='I-V curve')
plt.title('IV-analysis')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.legend(loc='best')
plt.yscale('log')
#plt.plot(x, result.best_fit, 'r-', label='R-squared ={}'.format(R))




# Create empty lists
wavelengths_1, transmission_1 = [], []
wavelengths_2, transmission_2 = [], []
wavelengths_3, transmission_3 = [], []
wavelengths_4, transmission_4 = [], []
wavelengths_5, transmission_5 = [], []
wavelengths_6, transmission_6 = [], []

#DCBias에 따른 값 가져오기
for measurement in root.iter('WavelengthSweep'):
    if measurement.get('DCBias') == '-2.0':
        wavelengths_1 = list(map(float, measurement.find('L').text.split(',')))
        transmission_1 = list(map(float, measurement.find('IL').text.split(',')))
    elif measurement.get('DCBias') == '-1.5':
        wavelengths_2 = list(map(float, measurement.find('L').text.split(',')))
        transmission_2 = list(map(float, measurement.find('IL').text.split(',')))
    elif measurement.get('DCBias') == '-1.0':
        wavelengths_3 = list(map(float, measurement.find('L').text.split(',')))
        transmission_3 = list(map(float, measurement.find('IL').text.split(',')))
    elif measurement.get('DCBias') == '-0.5':
        wavelengths_4 = list(map(float, measurement.find('L').text.split(',')))
        transmission_4 = list(map(float, measurement.find('IL').text.split(',')))
    elif measurement.get('DCBias') == '0':
        wavelengths_5 = list(map(float, measurement.find('L').text.split(',')))
        transmission_5 = list(map(float, measurement.find('IL').text.split(',')))
    elif measurement.get('DCBias') == '0.5':
        wavelengths_6 = list(map(float, measurement.find('L').text.split(',')))
        transmission_6 = list(map(float, measurement.find('IL').text.split(',')))

# scatter plot 그리기
plt.subplot(1, 2, 2)
plt.scatter(wavelengths_1, transmission_1, label='-2.0V')
plt.scatter(wavelengths_2, transmission_2, label='-1.5V')
plt.scatter(wavelengths_3, transmission_3, label='-1.0V')
plt.scatter(wavelengths_4, transmission_4, label='-0.5V')
plt.scatter(wavelengths_5, transmission_5, label='0V')
plt.scatter(wavelengths_6, transmission_6, label='-0.5V')
plt.title('Transmission spectra - as measured')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission (dB)')
plt.legend(loc='best')

plt.show()




# 그래프 저장
#plt.savefig('이름.png', dpi=300, bbox_inches='tight')
#plt.show()

