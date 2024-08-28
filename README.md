# NASA-Breath-Diagnostics-Challenge
https://bitgrit.net/competition/22
The objective of this challenge is to develop a diagnostic model by using NASA E-Nose data gathered from exhaled breath of 63 volunteers in a COVID-19 study.  Challenge participants will use advanced data preparation and AI techniques to overcome the limited sample size of subjects in the COVID-19 study.


The data consists of 63 txt files representing the 63 patients, numbered 1 to 63.
Each file contains the Patient ID, the COVID-19 Diagnosis Result (POSITIVE or NEGATIVE) and numeric measurements for 64 sensors, D1 to D64. These sensors are installed within the E-Nose device, and they each measure different molecular signals in the breath of the patients.

All sensor data is indexed by a timestamp with the format Min:Sec, which represents the minute of the hour, and the second of that minute in which that sensor was sampled. The hour of the timestamp has been left out, but when the minute counter resets, it means that the next hour has begun. Keep this in mind when working with this time axis.

In order to achieve maximum consistency across patients, the data was exposed to the E-Nose device using a pulsation bag that had previously collected a patient's breath. The E-Nose measurement procedure also includes flushing the sensors with ambient air, which can be used to calibrate the readings taken when exposed to human breath.

The data was exposed to the E-Nose device for all patients using windows of exposure through the following process:

1. 5 min baseline measurement using ambient air
2. 1 min breath sample exposure and
    measurement, using the filled breath bag
3. 2 min sensor “recovery” using ambient air
4. 1 min breath sample exposure and
    measurement, using the filled breath bag
5. 2 min sensor “recovery” using ambient air
6. 1 min breath sample exposure and
    measurement, using the filled breath bag
7. 2 min sensor “recovery” using ambient air
Total time = 14 mins

The data is distributed into training and test sets:
Train: 45 patients
Test: 18 patients
