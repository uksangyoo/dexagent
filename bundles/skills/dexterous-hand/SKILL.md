---
name: dexterous-hand
description: Compose measured finger-contact transitions and object-frame motions for a UR5e with a 20-DOF Tesollo DG5F-S through a DexterousConnector.
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [dexterity, contact, industrial]
gap:
  allowed_tools: [hand.observe, hand.schema, hand.assign_roles, hand.validate_action, hand.transition_contact, hand.move_object, hand.last_report]
  canonical_scripts:
    - move_object: scripts/move_object.py
    - transition_contact: scripts/transition_contact.py
  references:
    - title: Typed action contract
      path: references/action.schema.json
    - title: Contact transition contract
      path: references/transition.schema.json
  tools:
    hand.observe: Measured five-finger state, contact forces, slip, and object-relative motion.
    hand.assign_roles: Assign all five intended roles without changing measured contacts.
    hand.transition_contact: Execute contact establishment, maintenance, release, or receiver-first support transfer.
    hand.move_object: Execute an object-frame effect through guarded contact-point IK.
    hand.validate_action: Check the typed action and measured contact prerequisites without movement.
    hand.last_report: Retrieve the last structured execution result.
  errors: [infeasible, insufficient_support, force_unavailable, force_limit, self_collision, contact_lost, slip, object_dropped, timeout, ik_failure, unsupported]
---

Use this bundle with `dexagent.gap.connector.DexterousConnector`. The connector owns and registers `hand.*`; this bundle supplies agent guidance and canonical script adapters, without registering duplicate tools or another graph executor.

Call `hand.schema` for complete typed contracts and units. Quaternions are scalar-first `wxyz`, distances are meters, and forces are newtons. Object effects use `object` or `fixture` frames. For fixture-frame rotations the axis uses fixture coordinates and the pivot is the object's current origin.

Roles describe intent. Contacts describe measured physics. Assigning SUPPORT is not evidence of support. Read force and slip observations before moving the object or releasing a finger. `max_normal_force_N` is the sum of normal loads over a finger's contact patches. Supply a simulation-calibrated limit; the example's 5 N is not a validated hardware limit.

Compose operations as ordinary GaP tool nodes. Route the returned `status` (`succeeded` or `failed`), retain `report`, and use physical checkpoints over `world.sim_state['dexterous_hand']`. A report includes the intended action, actual contact changes, pose errors, feasibility, failed constraints, and repair hints. `hand.validate_action` checks contact/frame prerequisites; it is not a proof that an entire nonlinear trajectory is reachable.

For support transfer, establish and verify the receiving contact before releasing the donor. If establishment fails, keep the donor loaded. Expand a failed object action into explicit contact operations or smaller motions, and change roles/ordering based on the measured failure. Persist a reusable graph only after its operations and GaP physical checkpoints pass.

Sliding and rolling **observation** is available. Their execution controllers return `unsupported`. Cable feed/wrap commands are typed but return `unsupported` until deformable cable state and calibrated tension control are attached. The current revolute-nut curriculum does not establish free-object, threading, wrench, or cable benchmark performance. Do not repair an unsupported capability by inventing joint trajectories or declaring task success.
