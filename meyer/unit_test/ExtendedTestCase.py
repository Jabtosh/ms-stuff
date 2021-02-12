import difflib
import pprint
from unittest import TestCase


class ExtendedTestCase(TestCase):

    def assertSeqAlmostEqual(self, seq1, seq2, places=None, msg=None, delta=None):
        """An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed and has a length.

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            places: Number of decimals to match.
            delta: Maximum difference. Supersedes places.
            msg: Optional message to use on failure instead of a list of
                    differences.
        """
        seq_type_name = type(seq1).__name__

        differing = None
        try:
            len1 = len(seq1)
        except (TypeError, NotImplementedError):
            differing = f'First {seq_type_name} has no length. Non-sequence?'

        if differing is None:
            try:
                len2 = len(seq2)
            except (TypeError, NotImplementedError):
                differing = f'Second {seq_type_name} has no length. Non-sequence?'

        if differing is None:
            if seq1 == seq2:
                return

            differing = ''

            for i in range(min(len1, len2)):
                try:
                    item1 = seq1[i]
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('\nUnable to index element %d of first %s\n' % (i, seq_type_name))
                    break

                try:
                    item2 = seq2[i]
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('\nUnable to index element %d of second %s\n' % (i, seq_type_name))
                    break

                try:
                    self.assertAlmostEqual(item1, item2, places=places, delta=delta, msg=msg)
                except AssertionError as e:
                    differing += f'\nFirst differing element {i}:\n{e}\n'
                    break
            else:
                if len1 == len2:
                    return

            if len1 > len2:
                differing += f'\nFirst {seq_type_name} contains {len1 - len2} additional elements.\n'
            elif len1 < len2:
                differing += f'\nSecond {seq_type_name} contains {len2 - len1} additional elements.\n'

        standardMsg = differing
        diffMsg = '\n' + '\n'.join(
            difflib.ndiff(pprint.pformat(seq1).splitlines(),
                          pprint.pformat(seq2).splitlines()))

        standardMsg = self._truncateMessage(standardMsg, diffMsg)
        msg = self._formatMessage(msg, standardMsg + diffMsg)
        self.fail(msg)
