"""Physics class.

Same as MOOG.physics.Physics except doesn't use forces and doesn't update sprite
positions from velocity, since those are all done by the graph.
"""

from moog import physics


class Physics(physics.AbstractPhysics):
    """Force physics class."""

    def __init__(self, corrective_physics=()):
        """Constructor.

        Args:
            corrective_physics: Optional instance (or iterable of instances) of
                physics.AbstractPhysics to be applied every step before updating
                the sprite positions. This is typically used to apply
                corrections to sprite velocities. For example, it can be used to
                enforce rigid tethers between sprites.
        """
        super(Physics, self).__init__(updates_per_env_step=1)
        if not isinstance(corrective_physics, (list, tuple)):
            corrective_physics = [corrective_physics]
        self._corrective_physics = corrective_physics
    
    def reset(self, state):
        for corrective_physics in self._corrective_physics:
            corrective_physics.reset(state)

    def apply_physics(self, state, updates_per_env_step):
        """Move the sprites according to the physics."""

        for corrective_physics in self._corrective_physics:
            corrective_physics.apply_physics(state, updates_per_env_step)