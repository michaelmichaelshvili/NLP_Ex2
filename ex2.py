from collections import defaultdict
import re
import math
from random import choices


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should suppport the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.error_tables = None
        self.count_table = defaultdict(
            int)  # dict. count the letters (singled and pairs) in the text the modek build from
        self.vocabulary = set()  # the words n the text
        if lm is not None:
            self.update_from_lm()

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """
        lm = Ngram_Language_Model(n=n)
        lm.build_model(text)
        return lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM disctionary if set)

            Args:
                lm: a language model object
        """
        self.lm = lm
        self.update_from_lm()

    def update_from_lm(self):
        """
        Build vocabulary and count table for one/two letters. I decided to use the lm.get_model_dictionary to build the
        mentioned variables. Thereby, if we need to fix a word that not appear in the model, I consider this word as OOV.
        """
        self.vocabulary = set()
        self.count_table = defaultdict(int)
        for ngram, count in self.lm.get_model_dictionary().items():
            words = ngram.split(' ')
            for word in words:
                self.vocabulary.add(word)
            word = '#' + words[0]
            if len(word) == 0:
                continue
            for l in range(len(word) - 1):
                self.count_table[word[l]] += count
                self.count_table[word[l:l + 2]] += count
                if l == len(word) - 2:
                    self.count_table[word[l + 1]] += count

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in:
        <'deletion', 'insertion', 'transposition', 'substitution'> and the
        inner dict {str: int} represents the confution matrix of the
        specific errors, where str is a string of two characters mattching the
        row and culumn "indixes" in the relevant confusion matrix and the int is the
        observed count of such an error (computed from the specified errors file).
        Examples of such string are 'xy', for deletion of a 'y'
        after an 'x', insertion of a 'y' after an 'x'  and substitution
        of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.


        Notes:
            1. Ultimately, one can use only 'deletion' and 'insertion' and have
                'substitution' and 'transposition' derived. Again,  we use all
                four types explicitly in order to keep things simple.
        Args:
            errors_file (str): full path to the errors file. File format, TSV:
                                <error>    <correct>


        Returns:
            A dictionary of confusion "matrices" by error type (dict).
    """
        lines = []
        error_table = {'insertion': defaultdict(int), 'deletion': defaultdict(int), 'substitution': defaultdict(int),
                       'transposition': defaultdict(int)}
        with open(errors_file, 'r') as f:
            lines = [line.rstrip().lower().split('\t') for line in f.readlines()]
        err_type = None
        for line in lines:
            error, correct = line
            if len(error) > len(correct):  # insertion error
                err_type = 'insertion'
                err_letters, err_count = self.get_insertion_deletion_error(correct, error)
            elif len(error) < len(correct):  # deletion error
                err_type = 'deletion'
                err_letters, err_count = self.get_insertion_deletion_error(error, correct)
            else:  # substitution or transportation
                err_type, err_letters, err_count = self.get_trans_sub_error(error, correct)
            for i in range(len(err_letters)):
                error_table[err_type][err_letters[i]] = error_table[err_type].get(err_letters[i], 0) + err_count[i]
        return error_table

    def get_insertion_deletion_error(self, error, correct):
        """
        Find the error in the error word

        Args:
            error (str): the word we know is wrong
            correct (str): the correct word
        :return: the errors letters and count for each error
        """
        error_idx = None
        for i in range(len(error)):
            if error[i] != correct[i]:
                error_idx = i
                break
        if error_idx is None:  # error in last letter
            error_idx = len(correct) - 1
        err_count = [1]
        err_letters = []
        prev_letter = '#' if error_idx == 0 else correct[error_idx - 1]
        err_letters.append(prev_letter + correct[error_idx])
        dup_once = False
        for j in range(error_idx - 1, -1,
                       -1):  # if the worng letter is in series of the same letter. e.g. employee against employeee
            if correct[j] == correct[
                j + 1]:  # as long as the current letter is like the worng letter, add 1 to error count
                dup_once = True
                if j - 1 >= 0 and correct[j] == correct[j - 1]:
                    err_count[0] += 1
                elif j == 0:
                    err_letters.append('#' + correct[error_idx])
                    err_count.append(1)
            elif dup_once:  # only if there is at least one duplicate letter (employ*ee*) so 'ye' is also count as candidate mistake
                err_letters.append(correct[j] + correct[error_idx])
                err_count.append(1)
                break
            else:
                break

        return err_letters, err_count

    def get_trans_sub_error(self, error, correct):
        """
        Find the incorrect letters. the possible errors are substitution and transposition
        Args:
            error (str): the word we know is wrong
            correct (str): the correct word
        :return: the errors letters and count for each error
        :return:
        """
        error_idx = None
        err_type = None
        err_letters = None
        for i in range(len(error) - 1):
            if error[i] != correct[i]:
                if error[i + 1] == correct[i] and correct[i + 1] == error[i]:  # check if there is cross
                    err_type = 'transposition'
                    err_letters = correct[i:i + 2]
                else:
                    err_type = 'substitution'
                    err_letters = error[i] + correct[i]
                break
        if err_type is None:  # the error is substitution in last word
            err_type = 'substitution'
            err_letters = error[-1] + correct[-1]
        return err_type, [err_letters], [1]

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        self.error_tables = error_tables

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model is the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        text = normalize_text(text)
        words = text.split(' ')
        words_to_fix = words
        max_sentence, max_prob = None, float('-inf')
        new_words = None
        for idx, word in enumerate(
                words):  # if there are oov word, try fix only it (assumption there is only one error in sentence)
            if word not in self.vocabulary:
                new_words = [word]
                oov_word_idx = idx
                alpha = 1e-32
                words_to_fix = new_words
                break
        for idx, word in enumerate(words_to_fix):
            optional_edits = self.get_word_edits(word)
            for word, word_edit_prob in optional_edits:
                if new_words is not None:
                    new_sentence = words[:oov_word_idx] + [word] + words[oov_word_idx + 1:]
                else:
                    new_sentence = words[:idx] + [word] + words[idx + 1:]
                new_sentence_prob = self.calc_sentence_prob(' '.join(new_sentence),
                                                            math.log(1 - alpha) + math.log(word_edit_prob))
                if new_sentence_prob > max_prob:
                    max_prob = new_sentence_prob
                    max_sentence = new_sentence
        original_sentence_prob = self.calc_sentence_prob(text, math.log(alpha))
        if original_sentence_prob > max_prob:
            max_sentence = text.split(' ')
        return ' '.join(max_sentence)

    def calc_sentence_prob(self, sentence, channel_prob):
        """
        Calculate the probabily of given sentence to be generate with the noisy channel pros
        :param sentence: sentence to calculate its prob
        :param channel_prob: the prob of the changed word
        :return: the log prob of the sentence
        """
        prior_sentence_prob = self.evaluate(sentence)
        return prior_sentence_prob + channel_prob

    def get_word_edits(self, word):
        """
        build all edits for a given word. whitespace is enabled to be suggested as correction by check if the two words that
        it create are in vocabulary.
        :param word: word to get its edits
        :return: possible edits for given word. return only vocabulary edits.
        """
        letters = """abcdefghijklmnopqrstuvwxyz '"""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        insertions = [[L + R[1:], self.calc_edit_prob('insertion', (L[-1] if L else '#') + R[0], L[-1] if L else '#')]
                      for L, R in splits if R]
        transposes = [[L + R[1] + R[0] + R[2:], self.calc_edit_prob('transposition', R[1] + R[0], R[1] + R[0])] for L, R
                      in splits if len(R) > 1]
        replaces = [[L + c + R[1:], self.calc_edit_prob('substitution', R[0] + c, c)] for L, R in splits if R for c in
                    letters]
        deletions = [[L + c + R, self.calc_edit_prob('deletion', (L[-1] if L else '#') + c, (L[-1] if L else '#') + c)]
                     for L, R in splits for c in letters]
        edits = [[word, prob] for word, prob in (insertions + transposes + replaces + deletions) if
                 all([w in self.vocabulary for w in word.split()])]
        return self.edit_best_prob(edits)

    def calc_edit_prob(self, err_type, err_letters, letters_to_count):
        """
        Calculate the prob of edit to be the right edit. according to Kemighan 1990
        :param err_type (str): one of 'insertion', 'deletion', 'substitution' or 'transposition'
        :param err_letters (str): the letters that making the error
        :param letters_to_count (str): the letter(s) we count to normalize err_letters
        :return: the prob of edit to be the right edit
        """
        return (self.error_tables[err_type].get(err_letters, 0) + 1) / (self.count_table.get(letters_to_count, 0) + 1)

    def edit_best_prob(self, edits):
        """
        For case that word has same edits, save onlu the one with the best prob because it will get bigger score anyway.
        :param edits (list): edits of a word
        :return: the edit with the best prob
        """
        best_edit_prob = {}
        for edit, prob in edits:
            if edit not in best_edit_prob:
                best_edit_prob[edit] = prob
            else:
                if prob > best_edit_prob[edit]:
                    best_edit_prob[edit] = prob
        return [[edit, prob] for edit, prob in best_edit_prob.items()]


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Michael Michaelshvili', 'id': '318949443', 'email': 'michmich@post.bgu.ac.il'}


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a language model
        from a given text.
        It supoprts language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.n = n
        self.model_dict = defaultdict(
            int)  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.use_chars = chars
        self.V = None  # size of vocabulary
        self.context_by_n = None  # dict from context size to all contexts in this size
        self.contexts_prob = []

    def build_model(self, text):
        """populates the instance variable model_dict.

            Args:
                text (str): the text to construct the model from.
        """
        splitted = self.split(text)
        self.context_by_n = {i: defaultdict(int) for i in range(0, self.n)}

        for n in range(1, self.n + 1):
            for idx in range(len(splitted) - 1, n - 2, -1):
                ngram = self.get_ngram_by_last_index(splitted, idx, n)
                context, last_gram = self.split_context_gram(ngram)
                n_size_context = self.context_by_n[len(self.split(context))]
                if context in n_size_context:
                    if last_gram in n_size_context[context]:
                        n_size_context[context][last_gram] += 1
                    else:
                        n_size_context[context][last_gram] = 1
                else:
                    n_size_context[context] = {last_gram: 1}
                if n == self.n:
                    self.model_dict[ngram] += 1
        self.V = len(self.context_by_n[self.n - 1])
        sum_per_context = [sum(self.context_by_n[self.n - 1][context].values()) for context in
                           self.context_by_n[self.n - 1]]
        sum_all_context = sum(sum_per_context)
        self.contexts_prob = [context_sum / sum_all_context for context_sum in sum_per_context]

    def get_ngram_by_last_index(self, s, idx, n_size):
        """
        Return ngram of size self.n ends at index idx

            Args:
                s (list): list of tokens from which the ngram is extracted
                idx (int): the index from which to start build (reversed) the ngram
                n_size (int) the size of the ngram
            Retutns:
                ngram (str). the ngram was extracted

        """
        if n_size is None:
            n_size = self.n
        if self.use_chars:
            return "".join(s[max(idx - n_size + 1, 0): idx + 1])
        else:
            return " ".join(s[max(idx - n_size + 1, 0): idx + 1])

    def split_context_gram(self, ngram):
        """Return context and last gram from given ngram.

            Args:
                ngram (str): text to split

            Returns:
                context (str): all tokens except the last
                last_gram (str): last token
        """
        if self.use_chars:
            context, last_gram = ngram[:-1], ngram[-1]
        else:
            ngram_split = ngram.split(" ")
            context, last_gram = self.join(ngram_split[:-1]), ngram_split[-1]
        return context, last_gram

    def get_model_dictionary(self):
        """Returns the dictionary class object
        """
        return self.model_dict

    def get_model_window_size(self):
        """Returning the size of the context window (the n in "n-gram")
        """
        return self.n

    def join(self, tokens):
        """
        Returns tokens joined into one string. join according to self.use_chars

            Args:
                tokens (list): tokens to join

            Return:
                str. join text.
        """
        if self.use_chars:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)

    def split(self, text):
        """
        Returns tokens from text. the text is splitted according self.use_chars

            Args:
                text (str): string to split to tokens

            Return:
                List. Token list.
        """
        if self.use_chars:
            return list(text)
        else:
            if text == '':
                return []
            else:
                return text.split(' ')

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted. If the length of the specified context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """
        context_len = 0 if context is None else len(self.split(context))
        if context_len > n:  # context is longer than n - return the first n tokens
            return self.join(self.split(context)[:n])
        if context_len < self.n - 1:  # The context needs to be completed
            new_context = self.get_initial_context(context)
            if new_context is None:  # context is exhausted
                return context
            context = new_context

        final_text = self.split(context)
        if len(final_text) > n:
            return self.join(self.split(context)[:n])
        while len(final_text) < n:
            next_gram = self.get_next_gram(final_text)
            if next_gram is None:  # context is exhausted
                return self.join(final_text)
            final_text.append(next_gram)
        return self.join(final_text)

    def get_initial_context(self, init_context):
        """
            Sample the models' context distribution to get an initial context.
            If the initial context is exists, complete with the next grams we have see in the text.

            Args:
                init_context (str). The context that we got and need to evaluate from it
            Returns:
                String. An initial context

        """
        if init_context is None:
            return choices(list(self.context_by_n[self.n - 1].keys()), self.contexts_prob)[0]
        else:
            optional_context = [context for context in self.context_by_n[self.n - 1] if
                                context.startswith(init_context)]
            if len(optional_context) == 0:
                return None
            else:
                return choices(optional_context)[0]

    def get_next_gram(self, context):
        """
        Get the next gram according to context. the gram is chosen by weighted choice. The weight is number of apperance
        of the gram after the context.

        Returns:
            String. the chosen gram to be appropriate to the context.

        """
        if self.n == 1:
            last_context = ''
        else:
            last_context = self.join(context[-(self.n - 1):])
        last_context_len = len(self.split(last_context))
        if last_context not in self.context_by_n[self.n - 1]:
            return None
        possible_gram = self.context_by_n[self.n - 1][last_context].keys()
        possible_gram_probs_sum = sum(self.context_by_n[self.n - 1][last_context].values())
        possible_gram_probs = [self.context_by_n[self.n - 1][last_context][gram] / possible_gram_probs_sum for
                               gram in
                               possible_gram]
        return choices(list(self.context_by_n[self.n - 1][last_context].keys()), possible_gram_probs)[0]

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text to be a product of the model.
           Laplace smoothing should be applied if necessary.
           I chose to use stupid backoff. i.e calculate also kgrams where k<n, so their context is not of size n-1.
           I think that is more accurate method because the first grams are also important to understand the prob
           of the text to be evaluate by the model.
           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        splitted = self.split(text)
        prob = 0

        for i in range(len(splitted) - 1, -1, -1):
            ngram = self.get_ngram_by_last_index(splitted, i, None)
            context, last_gram = self.split_context_gram(ngram)
            context_len = len(self.split(context))
            if context in self.context_by_n[context_len] and last_gram in self.context_by_n[context_len][context]:
                prob += math.log(self.get_prob(context, last_gram))
            else:
                prob += math.log(self.smooth(context))
        return prob

    def get_prob(self, context, last_gram):
        """
        Calculate the probability to ngram. The probability the the last gram will be after the context.
        The calculation is the (number of occurrences of the ngram) / (number of occurrences of the context)

            Args:
                context (str): the context of the ngram we want to calculate its prob
                last_gram (str): the last_gram of the ngram we want to calculate its prob

            Returns:
                prob (float): the probability of the ngram to be evaluate.
        """
        context_len = len(self.split(context))
        c_ngram = self.context_by_n[context_len][context][last_gram]
        c_context = sum(self.context_by_n[context_len][context].values())
        return c_ngram / c_context

    def smooth(self, context):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                context (str): the context of the ngram we want to calculate its prob

            Returns:
                float. The smoothed probability.
        """
        c_context = 0
        context_len = len(self.split(context))
        if context in self.context_by_n[context_len]:
            c_context = sum(self.context_by_n[context_len][context].values())
        return 1 / (c_context + self.V)


def normalize_text(text, lower=True, pad_punc=True):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize
        lower (bool): if we want the model to learn only lower case tokens. Defaults to True.
        I chose to use this, because in English the first word in sentence is capital but has same meaning like th word
        without capital.
        pad_punc (bool): if treat punctuation like grams and padding them with white space. Defaults to True.
        I chose to use this because a dot or comma not supposed to influence the meaning.
      Returns:
        string. the normalized text.
    """
    normalized_text = text
    if lower:
        normalized_text = normalized_text.lower()
    if pad_punc:
        normalized_text = re.sub('(?<! )(?=[.,!?()!@#$%^&*\n\"])|(?<=[.,!?()!@#$%^&*\n\"])(?! )', r' ',
                                 normalized_text).strip()
    return normalized_text
