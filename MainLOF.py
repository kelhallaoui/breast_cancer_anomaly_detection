from ImportData import getData, separateDataOut, getPatientOut
from lof import LOF
from LOF_Methods import classified, getPrecision, classify240

leave_out = 2
patient_out = getPatientOut(2)

classification, decision, labels = classify240(patient_out)

print(getPrecision(decision, labels, patient_out))
