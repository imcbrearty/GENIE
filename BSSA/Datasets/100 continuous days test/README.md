### Data contents ###

## Pick files ##

P (picks): 
first column is time index from start of day (assuming 100 Hz sampling).
second column is station index (corresponding to indices of stations in the NC_EHZ_network.npz file).
third column is phase type (P waves, 0; S waves, 1), labeled by PhaseNet (these phase labels are not used by GENIE by default).
fourth column is maximum peak ground velocity from 1 s before to 2.5 s after each pick time.

sta_names_use: used stations on this day.

sta_ind_use: indices of stations used, corresponding to sta_names_use. Indices are the indices of stations in the NC_EHZ_network.npz network file.

day: the date of the file processed.

## Source files ##

