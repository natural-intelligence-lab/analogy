"""Conditions."""


class TestV0():
    """Single handcrafted maze."""

    _MAZE = [
        {
            'end': (0.5, 0.1),
            'directions': 'uluru',
            'lengths': [0.1, 0.1, 0.2, 0.1, 0.1],
        },
        {
            'end': (0.5, 0.1),
            'directions': 'ruluru',
            'lengths': [0.3, 0.3, 0.1, 0.2, 0.1, 0.1],
        },
    ]

    _PREY_ARM = 0
        
    def __call__(self):
        conditions = [
            [TestV0._MAZE, TestV0._PREY_ARM, {'name': 'TestV0'}],
        ]
        return conditions

