import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

# Approach the path
rootDir = "/Users/노정완/Desktop/대학교/23나노 1학기 PDF/공학프로그래밍2/data set/"
WorkingDir = rootDir + 'HY202103_D07_(0,0)_LION1_DCM_LMZC.xml'

# Parse XML file
tree = elemTree.parse(WorkingDir)
root = tree.getroot()  # 해당 트리의 root를 반환

# handle subplot
fig, (ax1, ax2) = plt.subplots(1, 2)

# graph 1
# Extract current and voltage data
for measurement in root.iter('IVMeasurement'):
    current_data = measurement.find('Current').text
    voltage_data = measurement.find('Voltage').text

# noinspection PyUnboundLocalVariable
current = list(map(float, current_data.split(',')))
# noinspection PyUnboundLocalVariable
voltage = list(map(float, voltage_data.split(',')))

# Adjust data to plot
current_abs = []
for i in range(len(current)):
    current_abs.append(abs(current[i]))
data_dict1 = {'voltage': voltage, 'current': current_abs}


# R-squared 구하기
def calc_R_squared():
    fp = np.polyfit(voltage, current_abs, 12)
    current_predicted_poly = np.polyval(fp, current_abs)
    residuals = current_abs - current_predicted_poly
    SSR = np.sum(residuals ** 2)
    SST = np.sum((y - np.mean(y) ** 2))
    return 1 - (SSR / SST)


# poly1d, numpy (np.poly1d)로 근사치 매기는 fitting
fp = np.polyfit(voltage, current_abs, 12)
f = np.poly1d(fp)
ax1.plot(voltage, f(voltage), linestyle='--', lw=2, color='r', label='best-fit')

# Plot data using matplotlib
scatter = ax1.scatter('voltage', 'current', data=data_dict1, color='mediumseagreen', label='data')

# Add annotations for current values
annotations = []
for x, y in zip(voltage, current_abs):
    if x in [-2.0, -1.5, -1, -0.5]:
        ann = ax1.annotate(f"{y:.2e}A", xy=(x, y), xytext=(0.5, -20), textcoords='offset points', ha='center',
                           fontsize=8)
    else:
        ann = ax1.annotate(f"{y:.2e}A", xy=(x, y), xytext=(0.5, 10), textcoords='offset points', ha='center',
                           fontsize=8)
    annotations.append(ann)

ax1.scatter(None, None, label=f"R² = {calc_R_squared()}")
ax1.set_yscale('log', base=10)
ax1.set_xlabel('Voltage [V]', size=16, fontweight='bold')
ax1.set_ylabel('Current [A]', size=16, fontweight='bold')
ax1.set_title('IV-analysis', size=20, fontweight='bold', style='italic')

ax1.tick_params(axis='both', which='major', size=14)  # tick 크기 설정
ax1.grid()
ax1.legend(fontsize=16)

# graph 2
# handle label color
cmap = plt.cm.get_cmap('jet')
a = 0

# Extract Wavelength and dB data
for measurement in root.iter('WavelengthSweep'):
    # choose a color for the scatter plot based on the iteration index
    color = cmap(a / 7)
    # get data from each element
    Wavelength_data = measurement.find('L').text
    Measured_transmission_data = measurement.find('IL').text
    # Convert the imported data into real numbers and lists to graph
    wavelength = list(map(float, Wavelength_data.split(',')))
    measured_transmission = list(map(float, Measured_transmission_data.split(',')))
    # make it a dict for easier handling
    data_dict2 = {'Wavelength': wavelength, 'Measured_transmission': measured_transmission}
    # Create a scatter plot using the data
    if not measurement == list(root.iter('WavelengthSweep'))[-1]:
        scatter = plt.scatter('Wavelength', 'Measured_transmission', data=data_dict2, color=color, s=3,
                              label=measurement.get('DCBias') + ' V')
    else:
        scatter = plt.scatter('Wavelength', 'Measured_transmission', data=data_dict2, color=color, s=3, label=None)
    # increment the color index
    a += 1

# Plot data using matplotlib
# plt.plot('voltage', 'current', data=data_dict, color='mediumseagreen', marker='o')
ax2.set_xlabel('Wavelength [nm]', size=16, fontweight='bold')
ax2.set_ylabel('Measured_transmission [dB]', size=16, fontweight='bold')
ax2.set_title('Transmission spectra - as measured', size=20, fontweight='bold', style='italic')

ax2.tick_params(axis='both', which='major', size=14)  # tick 크기 설정
ax2.grid()
ax2.legend(loc='lower center', ncol=3, fontsize=14)

plt.show()
