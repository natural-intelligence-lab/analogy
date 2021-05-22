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

    DIRECTIONS = [
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([0, -1]),
        np.array([0, 1]),
        np.array([0, 0]),
    ]

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
            return GridActions.DIRECTIONS[self._current_key]
        else:
            return GridActions.DIRECTIONS[4]  # Zero action


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


class HandEye():
    """Tkinter frame for composite Grid and SetPosition action space."""
    
    def __init__(self,
                 root,
                 canvas,
                 canvas_half_width,
                 hand_key,
                 eye_key):
        """Constructor.

        Args:
            root: Instance of tk.Frame. Root frame in which the gui frame lives.
            canvas: Canvas object to add position-setting functionality to.
            canvas_half_width: Int. Half-width of the canvas.
            hand_key: String. Key of the hand action space (arrow keys).
            eye_key: String. Key of the eye action space (mouse).
        """
        self._hand_key = hand_key
        self._eye_key = eye_key

        self._hand = GridActions(root=root, canvas_half_width=canvas_half_width)
        self._eye = SetPositionFrame(
            canvas=canvas, canvas_half_width=canvas_half_width)

        self.canvas = self._hand.canvas
        
    @property
    def action(self):
        """Return the composite action."""
        action = {
            self._hand_key: self._hand.action,
            self._eye_key: self._eye.action,
        }
        return action
