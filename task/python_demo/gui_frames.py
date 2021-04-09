"""This file contains GUI frames for human_agent.py.

The classes in this file are interfaces for playing the demo with different
action spaces. If you are using an action space that doesn't fall into one of
these categories, you must create your own gui for it.

Note: on some computers, the keyboard interfaces don't work properly, and
holding letter or arrow keys repeatedly takes actions. This is an issue with
your computer's keyboard sending rapid press/release signals when you hold down
a key. You can resolve this by (i) changing your computer's keyboard settings to
not do this behavior when you hold down a key, or (ii) modify the class in this
file that you're using to do whatever behavior you want when a key is held down
(this will involve a bit of debugging for you).
"""

import numpy as np
import tkinter as tk


class GridActions(tk.Frame):
    """Grid actions Tkinter frame.

    This creates an empty Tkinter frame where the joystick would be. It also
    registers bindings responding to arrow key presses and releases, and turns
    them into discrete actions for a Grid action space.
    """

    def __init__(self, root, canvas_half_width=100):
        """Constructor.

        Args:
            root: Instance of tk.Frame. Root frame in which the gui frame lives.
            canvas_half_width: Int. Half of the width of the canvas to create.
        """
        super(GridActions, self).__init__(root)
        self._current_key = 4  # Do-nothing action

        # Create a canvas
        self.canvas = tk.Canvas(
            width=2 * canvas_half_width,
            height=2 * canvas_half_width)

        # Add bindings for key presses and releases
        root.bind('<KeyPress>', self._key_press)
        root.bind('<KeyRelease>', self._key_release)

    def _get_action_from_event(self, event):
        if event.keysym == 'Left':
            return 0
        elif event.keysym == 'Right':
            return 1
        elif event.keysym == 'Down':
            return 2
        elif event.keysym == 'Up':
            return 3
        else:
            return None

    def _key_press(self, event):
        self._current_key = self._get_action_from_event(event)

    def _key_release(self, event):
        if self._get_action_from_event(event) == self._current_key:
            self._current_key = None

    @property
    def action(self):
        if self._current_key is not None:
            return self._current_key
        else:
            return 4  # Do-nothing action


class SetPositionFrame():
    """SetPosition Tkinter frame.

    This creates position_setting functionality for a canvas. Typically this is
    done to the HumanAgent._env_canvas.
    """
    def __init__(self, canvas, canvas_half_width):
        """Constructor.

        Args:
            canvas: Canvas object to add position-setting functionality to.
            canvas_half_width: Int. Half-width of the canvas.
        """
        # Add bindings for clicking, dragging and releasing the joystick
        canvas.bind('<ButtonPress-1>', self._mouse_press)
        canvas.bind('<ButtonRelease-1>', self._mouse_release)
        canvas.bind('<B1-Motion>', self._mouse_move)

        self._canvas_half_width = canvas_half_width
        self._mouse_is_pressed = False
        self._mouse_coords = np.array([0.5, 0.5])

    def _mouse_press(self, event):
        self._place_mouse(event)
        self._mouse_is_pressed = True

    def _mouse_release(self, event):
        self._mouse_is_pressed = False

    def _mouse_move(self, event):
        if self._mouse_is_pressed:
            self._place_mouse(event)

    def _place_mouse(self, event):
        """Place the self._mouse_coords (x, y) coordinates of a mouse event."""
        centered_event_coords = (
            np.array([event.x, event.y], dtype=float) - self._canvas_half_width)
        centered_event_coords = np.clip(
            centered_event_coords,
            -self._canvas_half_width,
            self._canvas_half_width,
        )
        self._mouse_coords = 0.5 * (
            1 + centered_event_coords.astype(float) / self._canvas_half_width)

    @property
    def action(self):
        """Return the mouse's position as an action in [0, 1] x [0, 1]."""
        return np.array([self._mouse_coords[0], 1. - self._mouse_coords[1]])


class GridSetPosition():
    """Tkinter frame for composite Grid and SetPosition action space."""
    
    def __init__(self,
                 root,
                 canvas,
                 canvas_half_width,
                 grid_key,
                 set_position_key):
        """Constructor.

        Args:
            root: Instance of tk.Frame. Root frame in which the gui frame lives.
            canvas: Canvas object to add position-setting functionality to.
            canvas_half_width: Int. Half-width of the canvas.
            grid_key: String. Key of the Grid action space.
            set_position_key: String. Key of the SetPosition action space.
        """
        self._grid_key = grid_key
        self._set_position_key = set_position_key

        self._grid = GridActions(root=root, canvas_half_width=canvas_half_width)
        self._set_position = SetPositionFrame(
            canvas=canvas, canvas_half_width=canvas_half_width)

        self.canvas = self._grid.canvas
        
    @property
    def action(self):
        """Return the composite action."""
        action = {
            self._grid_key: self._grid.action,
            self._set_position_key: self._set_position.action,
        }
        return action


class JoystickFrame(tk.Frame):
    """Joystick Tkinter frame.

    This creates the frame for an interactive joystick. The joystick consists of
    three objects:
        (i) A large gray "motion zone" circle in the background. This is the
            area in which the joystick can be moved.
        (ii) A small black "center point" circle fixed in the middle ground.
            This indicates the joystick position that give zero action.
        (iii) A green "joystick" circle in the foreground that can be moved. The
            center of this circle is the action readout.

    If the mouse is not clicked, then position of the joystick centerpoint is 0
    (in the center of the motion zone). If the mouse is currently pressed, then
    the position of the joystick is the closest point in the motion zone to the
    mouse position. Namely, if the mouse is in the motion zone then the joystick
    is directly under the mouse, and if the mouse is not in the motion zone then
    the joystick is at the edge of the motion zone closest to the mouse.

    Thus the joystick can be moved by clicking the mouse anywhere (in which case
    the joystick jumps to that position) and dragging the pressed mouse around
    (in which case the joystick moves underneath the mouse).
    """

    def __init__(self, root, canvas_half_width=100, motion_zone_radius=90,
                 joystick_radius=10, center_point_radius=3):
        """Constructor.

        Args:
            root: Instance of tk.Frame. Root frame in which the joystick lives.
            canvas_half_width: Int. Half of the width of the canvas on which the
                joystick is rendered.
            motion_zone_radius: Int. Radius of the motion zone.
            joystick_radius: Int. Radius of the joystick.
            center_point_radius: Int. Radius of the center point.
        """
        super(JoystickFrame, self).__init__(root)

        self._joystick_radius = joystick_radius
        self._canvas_half_width = canvas_half_width
        self._motion_zone_radius = motion_zone_radius
        self._center_point_radius = center_point_radius

        # Create a canvas
        self.canvas = tk.Canvas(
            width=2 * canvas_half_width,
            height=2 * canvas_half_width)

        # Create motion zone, center point, and joystick
        self._create_items()

        # Add bindings for clicking, dragging and releasing the joystick
        self.canvas.bind('<ButtonPress-1>', self._mouse_press)
        self.canvas.bind('<ButtonRelease-1>', self._mouse_release)
        self.canvas.bind('<B1-Motion>', self._mouse_move)

        self._mouse_is_pressed = False

    def _create_items(self):
        # Create motion zone
        self.canvas.create_oval(
            self._canvas_half_width - self._motion_zone_radius,
            self._canvas_half_width - self._motion_zone_radius,
            self._canvas_half_width + self._motion_zone_radius,
            self._canvas_half_width + self._motion_zone_radius,
            fill='gray80',
        )

        # Create center point
        self.canvas.create_oval(
            self._canvas_half_width - self._center_point_radius,
            self._canvas_half_width - self._center_point_radius,
            self._canvas_half_width + self._center_point_radius,
            self._canvas_half_width + self._center_point_radius,
            fill='black',
        )

        # Create joystick
        self.joystick = self.canvas.create_oval(
            self._canvas_half_width - self._joystick_radius,
            self._canvas_half_width - self._joystick_radius,
            self._canvas_half_width + self._joystick_radius,
            self._canvas_half_width + self._joystick_radius,
            fill='green',
        )

    def _recenter_joystick(self):
        """Move the joystick to the center."""
        new_coords = [
            self._canvas_half_width - self._joystick_radius,
            self._canvas_half_width - self._joystick_radius,
            self._canvas_half_width + self._joystick_radius,
            self._canvas_half_width + self._joystick_radius,
        ]
        self.canvas.coords(self.joystick, new_coords)

    def _place_joystick(self, event):
        """Place the joystick near the (x, y) coordinates of a mouse event.

        If the event (x, y) is inside the motion zone, the joystick is placed
        directly at that position. If it is outside the motion zone, the
        joystick is placed on the edge of the motion zone nearest to that point.
        """
        centered_event_coords = (
            np.array([event.x, event.y], dtype=float) - self._canvas_half_width)
        event_dist = np.linalg.norm(centered_event_coords)
        rescale_factor = min(1, self._motion_zone_radius / event_dist)
        centered_event_coords *= rescale_factor
        event_coords = centered_event_coords + self._canvas_half_width
        event_coords = event_coords.astype(int)

        new_coords = [
            event_coords[0] - self._joystick_radius,
            event_coords[1] - self._joystick_radius,
            event_coords[0] + self._joystick_radius,
            event_coords[1] + self._joystick_radius,
        ]
        self.canvas.coords(self.joystick, new_coords)

    def _mouse_press(self, event):
        self._place_joystick(event)
        self._mouse_is_pressed = True

    def _mouse_release(self, event):
        self._mouse_is_pressed = False
        self._recenter_joystick()

    def _mouse_move(self, event):
        if self._mouse_is_pressed:
            self._place_joystick(event)

    @property
    def action(self):
        """Return the joystick's position as an action in [-1, 1] x [-1, 1]."""
        joystick_coords = self.canvas.coords(self.joystick)
        joystick_center = np.array([
            joystick_coords[0] + self._joystick_radius,
            joystick_coords[1] + self._joystick_radius
        ])
        joystick_center -= self._canvas_half_width
        action = joystick_center.astype(float) / self._motion_zone_radius
        return np.array([action[0], -1 * action[1]])
