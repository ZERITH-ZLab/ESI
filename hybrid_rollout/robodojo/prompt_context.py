"""Version of teacher-visible evaluation guidance, independent of model/source."""

CONTEXT_VERSION = 'v3'

# Public task semantics only: no layout truth, score queries or action script.
TASK_NOTES = {
    'deposit_coin': 'Pick up the coin from its holder, insert it into the coin bank, and return the robot arms to their initial poses.',
    'organize_table': 'Place the four named desk items at their instructed destinations, let them settle, and return the arms.',
    'classify_objects_by_language': 'Follow this episode\'s language-specified category-to-basket mapping for every object.',
    'imitate_sorting_sequence': 'Observe and remember the demonstration before placing the corresponding objects in the same order.',
    'arrange_largest_number': 'Arrange the digits left to right as the largest number, correctly oriented and placed on their pads.',
    'pack_objects_into_box': 'Place every object inside the box with its front facing left, as instructed.',
    'classify_objects': 'Group all objects by category into the three baskets.',
    'build_tower': 'Build the supported tower layers using the wooden blocks and boards.',
    'make_kong': 'Wait for the discard, expose the matching tiles, and finish the tile-handling sequence without disturbing the other tiles.',
    'fold_clothes': 'Fold the sleeves inward and the lower garment upward into a neat folded shape.',
    'put_bottles_into_dustbin': 'Put all bottles inside the dustbin.',
}


def task_context(task):
    base = task.removesuffix('_random')
    if base not in TASK_NOTES:
        return {}
    return dict(process=TASK_NOTES[base], requires_arm_return=base != 'make_kong')
