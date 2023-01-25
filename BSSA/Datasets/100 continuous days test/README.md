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

cat: the USGS reported catalog with M>1.  
first column is latitude (degrees).  
second column is longitude (degrees).  
third column is depth (m, assuming 0 depth is sea level, with negative depths inside the Earth, and positive depths above sea level).  
fourth column is origin time (s, with time as value since start of day).  
fifth column is magnitude (this is either Ml, Md, or Mw magnitude type, depending on what USGS supplies for each event).  

srcs: the initial set of detected sources by GENIE.  
first column is latitude (degrees).  
second column is longitude (degrees).  
third column is depth (m, assuming 0 depth is sea level, with negative depths inside the Earth, and positive depths above sea level).  
fourth column is origin time (s, with time as value since start of day).  
fifth column is max value of GENIE output.  
(source locations in srcs are less well localized than srcs_trv).  

srcs_trv: the travel time located sources for each detected source by GENIE.  
first column is latitude (degrees).  
second column is longitude (degrees).  
third column is depth (m, assuming 0 depth is sea level, with negative depths inside the Earth, and positive depths above sea level). \\
fourth column is origin time (s, with time as value since start of day).  

mag_r: magnitudes of each event estimated from srcs locations (most stable).  

mag_trv: magnitudes of each event estimated from srcs_trv locations (slightly less stable than mag_r).  

izmatch1: indices of matched events between cat and srcs_r.  
first column are indices in cat.  
second column are indices in srcs_r.  

izmatch2: indices of matched events between cat and srcs_trv.  
first column are indices in cat.  
second column are indices in srcs_trv.  

locs_use: set of station locations used on this day.  

ind_use: set of station indices used on this day (corresponding to indices of stations in the NC_EHZ_network.npz file).  

Picks/<n>_Picks_P: subset of picks associated to event 0...(N - 1) for each event index <n>, declared as P waves, where N = len(srcs).  
first column is time of pick (s).  
second column is index of pick (corresponding to indices of stations in the NC_EHZ_network.npz file).  
third column is maximum peak ground velocity of pick from 1 s before to 2.5 s after each pick time.  
fourth column is pick likelihood value output by PhaseNet.  
fifth column is phase type assigned by GENIE (P waves, 0; S waves, 1).  
sixth column is source index in srcs.  

Picks/<n>_Picks_S: subset of picks associated to event 0...(N - 1) for each event index <n>, declared as S waves, where N = len(srcs).  
first column is time of pick (s).  
second column is index of pick (corresponding to indices of stations in the NC_EHZ_network.npz file).  
third column is maximum peak ground velocity of pick from 1 s before to 2.5 s after each pick time.  
fourth column is pick likelihood value output by PhaseNet.  
fifth column is phase type assigned by GENIE (P waves, 0; S waves, 1).  
sixth column is source index in srcs.  

Picks/<n>_Picks_P_perm: same as Picks/<n>_Picks_P, with the indices in second column corresponding to station index in locs_use, instead of the absolute set of indices in the NC_EHZ_network.npz file.  

Picks/<n>_Picks_S_perm: same as Picks/<n>_Picks_S, with the indices in second column corresponding to station index in locs_use, instead of the absolute set of indices in the NC_EHZ_network.npz file.  
