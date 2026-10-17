"""P13 re-check after the chunked random streams (sim4 chunk = 300 so that chunk boundaries fall inside the run and at
the interruption point): checkpoint/resume bit identity and A vs A_exact."""
import sim4
sim4.DEFAULT["chunk"] = 300
import p13_sim4_sanity as p
p.part_ckpt()
