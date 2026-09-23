"""
define the sessions for preprocessing here
"""

dataset_sessions = {
    # MSCL sessions
    # '2026-02-11 mouse c57 565 eMSCL A': dict(skip_triggers=[]),  # chirp done
    # '2026-02-16 mouse c57 566 eMSCL A': dict(skip_triggers=[]),  # chirp done
    # '2026-05-06 mouse c57 611 MscL A': dict(skip_triggers=[]),  # chirp done
    # '2026-08-25 mouse c57 754 eMSCL A' : dict(skip_triggers=[]),
    # '2026-08-25 mouse c57 754 eMSCL B' : dict(skip_triggers=[]),
    # '2026-08-25 mouse c57 754 eMSCL C' : dict(skip_triggers=[]),
    # '2026-08-26 mouse c57 755 eMSCL B' : dict(skip_triggers=[]),
    # '2026-08-26 mouse c57 755 eMSCL C': dict(skip_triggers=[]),

    # Mekano sessions
    # '2026-02-19 mouse c57 5713 Mekano6 A': dict(skip_triggers=[]),  # chirp done
    # '2026-03-25 mouse c57 617 Mekano6 B': dict(skip_triggers=[]),  # chirp done
    # '2026-05-13 mouse c57 615 Mekano6 A': dict(skip_triggers=[]),  # chirp done
    # '2026-06-12 mouse c57 649 Mekano6 C': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),  # PC home
    # '2026-06-12 mouse c57 649 Mekano6 D': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),  # PC home
    # '2026-06-16 mouse c57 645 Mekano6 B': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),  # problem in pipe
    # '2026-06-16 mouse c57 645 Mekano6 C': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),   # chirp done
    # '2026-06-30 rat LE 803 Mekano6 A': dict(skip_triggers=[]), # chirp done
    # '2026-06-30 rat LE 803 Mekano6 B': dict(skip_triggers=[5]),  #  chirp done
    # '2026-06-30 rat LE 803 Mekano6 C': dict(skip_triggers=[]),  # chirp done
    # '2026-07-02 mouse c57 650 Mekano6 A': dict(skip_triggers=[]),  # chirp done
    # '2026-07-02 mouse c57 650 Mekano6 B': dict(skip_triggers=[7]),  # chirp done
    # '2026-07-08 rat LE 3322 Mekano6 A': dict(skip_triggers=[]),  # chirp done
    # '2026-07-08 rat LE 3322 Mekano6 B': dict(skip_triggers=[],
    #                                          align_trials=True),  # TODO the CNPQX recording is not added
    # '2026-07-09 rat LE 0353 Mekano6 A': dict(skip_triggers=[]),  # chirp done
    # '2026-07-09 rat LE 0353 Mekano6 B': dict(skip_triggers=[]),  # chirp done
     # '2026-09-08 mouse c57 750 Mekano6 A': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),  # chirp done
    # '2026-09-08 mouse c57 750 Mekano6 B': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),  # chirp done
    # '2026-09-08 mouse c57 750 Mekano6 C': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128),  # chirp done
    # '2026-09-09 mouse c57 758 Mekano6 A': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128), # chirp done
    # '2026-09-09 mouse c57 758 Mekano6 C': dict(skip_triggers=[], laser_trigger_channel=255, dmd_trigger_channel=128), # chirp done
    # '2026-09-15 mouse c57 752 Mekano6 A': dict(skip_triggers=[]),  # chirp done
    # '2026-09-15 mouse c57 752 Mekano6 B': dict(skip_triggers=[]),  # chirp done, problem with chirp rec
    # '2026-09-15 mouse c57 752 Mekano6 C': dict(skip_triggers=[3, 4]), # chirp done, Problem with rec 3 and 4?

    # # Control (no virus)
    # '2026-07-01 mouse c57 653 NoVirus C': dict(skip_triggers=[1]),
    # '2026-07-21 rat LE 9999 NoVirus A': dict(skip_triggers=[1, 2]),
    # '2026-07-21 rat LE 9382 NoVirus B': dict(skip_triggers=[1, 2]),
    # '2026-07-21 mouse c57 647 NoVirus C': dict(skip_triggers=[1, 2]),
    # '2026-08-18 rat LE 3243 NoVirus B' : dict(skip_triggers=[1, 2]),
    # '2026-08-18 rat LE 3243 NoVirus C' : dict(skip_triggers=[1, 2]),  # skip, has only CNPQX?
    # '2026-09-09 mouse c57 758 NoVirus B': dict(skip_triggers=[1,2], laser_trigger_channel=255, dmd_trigger_channel=128),


    # Sessions without usefull data
    # '2026-05-06 mouse c57 611 MscL C': dict(skip_triggers=[6]),  #
    # '2026-03-17 mouse c57 613 eMSCL A': dict(skip_triggers=[1]),  # bad
    # '2026-03-17 mouse c57 613 eMSCL B': dict(skip_triggers=[1]),  # bad
    # '2026-03-24 mouse c57 616 Mekano6 A': dict(skip_triggers=[1]),  # no triggers
    # '2026-03-25 mouse c57 617 Mekano6 A': dict(skip_triggers=[1]),  # bad
    # '2026-03-25 mouse c57 617 Mekano6 C': dict(skip_triggers=[2, 3]),  # bad
    # '2026-03-31 mouse c57 622 Mekano4 A': dict(skip_triggers=[1, 2]),  # bad
    # '2026-04-01 mouse c57 623 Mekano4 4 A': dict(skip_triggers=[]),  # bad
    # '2026-04-08 mouse c57 621 Mekano4 A': dict(skip_triggers=[1, 2]),  # 2 is not a complete rec
    # '2026-04-08 mouse c57 621 Mekano4 B': dict(skip_triggers=[1])  # 2 is not a complete rec
    # '2026-04-14 mouse c57 612 eMSCL A': dict(skip_triggers=[1, 2]),  # bad
    # '2026-04-16 mouse c57 614 Mekano6 A': dict(skip_triggers=[1]),  # bad
    # '2026-04-16 mouse c57 614 Mekano6 B': dict(skip_triggers=[1]),  # bad
    # '2026-04-16 mouse c57 614 Mekano6 C': dict(skip_triggers=[1]),  # bad
    # '2026-08-26 mouse c57 755 eMSCL A' : dict(skip_triggers=[1, 2]),
}