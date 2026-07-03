"""
define the sessions for preprocessing here
"""

dataset_sessions = {
    # '2026-02-11 mouse c57 565 eMSCL A': dict(skip_triggers=[1]),  # good
    # '2026-02-16 mouse c57 566 eMSCL A': dict(skip_triggers=[]),  # good
    # '2026-02-19 mouse c57 5713 Mekano6 A': dict(skip_triggers=[1]),  # good
    # '2026-03-25 mouse c57 617 Mekano6 B': dict(skip_triggers=[]),  #
    # '2026-05-06 mouse c57 611 MscL A': dict(skip_triggers=[1]),  #
    # '2026-05-06 mouse c57 611 MscL C': dict(skip_triggers=[1, 5, 6]),  #
    # '2026-05-13 mouse c57 615 Mekano6 A': dict(skip_triggers=[1]),
    # '2026-06-12 mouse c57 649 Mekano6 C': dict(skip_triggers=[1], laser_trigger_channel=255, dmd_trigger_channel=128),  # good
    # '2026-06-12 mouse c57 649 Mekano6 D': dict(skip_triggers=[1], laser_trigger_channel=255, dmd_trigger_channel=128),
    # '2026-06-16 mouse c57 645 Mekano6 B': dict(skip_triggers=[1]),  # TODO (data at pc home)
    # '2026-06-16 mouse c57 645 Mekano6 C': dict(skip_triggers=[1]), laser_trigger_channel=255, dmd_trigger_channel=128),
    '2026-06-30 rat LE 803 Mekano6 A': dict(skip_triggers=[1]),
    '2026-06-30 rat LE 803 Mekano6 B': dict(skip_triggers=[1, 5]),
    '2026-06-30 rat LE 803 Mekano6 C': dict(skip_triggers=[1]),
    '2026-07-01 mouse c57 653 NoVirus C': dict(skip_triggers=[1]),
    '2026-07-02 mouse c57 650 Mekano6 A': dict(skip_triggers=[1]),



    # Sessions without usefull data
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
}