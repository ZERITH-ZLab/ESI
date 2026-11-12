# Gripper contact telemetry

The diagnostic contact API returns the cached, acknowledged action window. It does not advance physics or add a GPT tool. CPU PhysX is supported; unsupported backends and incomplete windows are marked invalid rather than reported as zero force.

Per-finger records contain contact actors, normal impulses and window aggregates. Reported force is normal impulse divided by the physics time step; it is not a friction-inclusive grasp load or a wrist force measurement. Opposing finger forces must not be summed to infer grasp strength.

`gripper_contact.jsonl` retains raw windows and step identities. Validity flags distinguish unavailable telemetry from valid windows without contact. Contact or position stability does not prove successful grasping or task completion.
