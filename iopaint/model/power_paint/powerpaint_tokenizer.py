import copy
import random
from typing import Any, List, Union
from transformers import CLIPTokenizer

from iopaint.schema import PowerPaintTask


def add_task_to_prompt(prompt, negative_prompt, task: PowerPaintTask):
    if task == PowerPaintTask.object_remove:
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    elif task == PowerPaintTask.context_aware:
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif task == PowerPaintTask.shape_guided:
        promptA = prompt + " P_shape"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif task == PowerPaintTask.outpainting:
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    else:
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt

    return promptA, promptB, negative_promptA, negative_promptB


def task_to_prompt(task: PowerPaintTask):
    promptA, promptB, negative_promptA, negative_promptB = add_task_to_prompt(
        "", "", task
    )
    return (
        promptA.strip(),
        promptB.strip(),
        negative_promptA.strip(),
        negative_promptB.strip(),
    )


class PowerPaintTokenizer:
    def __init__(self, tokenizer: CLIPTokenizer):
        self.wrapped = tokenizer
        self.token_map = {}
        placeholder_tokens = ["P_ctxt", "P_shape", "P_obj"]
        num_vec_per_token = 10
        for placeholder_token in placeholder_tokens:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                output.append(ith_token)
            self.token_map[placeholder_token] = output

    def __getattr__(self, name: str) -> Any:
        if name == "wrapped":
            return super().__getattr__("wrapped")

        try:
            return getattr(self.wrapped, name)
        except AttributeError:
            try:
                return super().__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "'name' cannot be found in both "
                    f"'{self.__class__.__name__}' and "
                    f"'{self.__class__.__name__}.tokenizer'."
                )

    def try_adding_tokens(self, tokens: Union[str, List[str]], *args, **kwargs):
        """Attempt to add tokens to the tokenizer.

        Args:
            tokens (Union[str, List[str]]): The tokens to be added.
        """
        num_added_tokens = self.wrapped.add_tokens(tokens, *args, **kwargs)
        assert num_added_tokens != 0, (
            f"The tokenizer already contains the token {tokens}. Please pass "
            "a different `placeholder_token` that is not already in the "
            "tokenizer."
        )

    def get_token_info(self, token: str) -> dict:
        """Get the information of a token, including its start and end index in
        the current tokenizer.

        Args:
            token (str): The token to be queried.

        Returns:
            dict: The information of the token, including its start and end
                index in current tokenizer.
        """
        token_ids = self.__call__(token).input_ids
        start, end = token_ids[1], token_ids[-2] + 1
        return {"name": token, "start": start, "end": end}

    def add_placeholder_token(
        self, placeholder_token: str, *args, num_vec_per_token: int = 1, **kwargs
    ):
        """Add placeholder tokens to the tokenizer.

        Args:
            placeholder_token (str): The placeholder token to be added.
            num_vec_per_token (int, optional): The number of vectors of
                the added placeholder token.
            *args, **kwargs: The arguments for `self.wrapped.add_tokens`.
        """
        output = []
        if num_vec_per_token == 1:
            self.try_adding_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                self.try_adding_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)

        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f"The tokenizer already has placeholder token {token} "
                    f"that can get confused with {placeholder_token} "
                    "keep placeholder tokens independent"
                )
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(
        self,
        text: Union[str, List[str]],
        vector_shuffle: bool = False,
        prop_tokens_to_load: float = 1.0,
    ) -> Union[str, List[str]]:
        """Replace the keywords in text with placeholder tokens. This function
        will be called in `self.__call__` and `self.encode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(
                    self.replace_placeholder_tokens_in_text(
                        text[i], vector_shuffle=vector_shuffle
                    )
                )
            return output

        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[: 1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return text

    def replace_text_with_placeholder_tokens(
        self, text: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """Replace the placeholder tokens in text with the original keywords.
        This function will be called in `self.decode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(self.replace_text_with_placeholder_tokens(text[i]))
            return output

        for placeholder_token, tokens in self.token_map.items():
            merged_tokens = " ".join(tokens)
            if merged_tokens in text:
                text = text.replace(merged_tokens, placeholder_token)
        return text

    def __call__(
        self,
        text: Union[str, List[str]],
        *args,
        vector_shuffle: bool = False,
        prop_tokens_to_load: float = 1.0,
        **kwargs,
    ):
        """The call function of the wrapper.

        Args:
            text (Union[str, List[str]]): The text to be tokenized.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(
            text, vector_shuffle=vector_shuffle, prop_tokens_to_load=prop_tokens_to_load
        )

        return self.wrapped.__call__(replaced_text, *args, **kwargs)

    def encode(self, text: Union[str, List[str]], *args, **kwargs):
        """Encode the passed text to token index.

        Args:
            text (Union[str, List[str]]): The text to be encode.
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(text)
        return self.wrapped(replaced_text, *args, **kwargs)

    def decode(
        self, token_ids, return_raw: bool = False, *args, **kwargs
    ) -> Union[str, List[str]]:
        """Decode the token index to text.

        Args:
            token_ids: The token index to be decoded.
            return_raw: Whether keep the placeholder token in the text.
                Defaults to False.
            *args, **kwargs: The arguments for `self.wrapped.decode`.

        Returns:
            Union[str, List[str]]: The decoded text.
        """
        text = self.wrapped.decode(token_ids, *args, **kwargs)
        if return_raw:
            return text
        replaced_text = self.replace_text_with_placeholder_tokens(text)
        return replaced_text
