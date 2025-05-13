import base64
import io
import json
import os
from pathlib import Path
from urllib.parse import unquote
from PIL import PngImagePlugin, _util, ImagePalette
from PIL import Image as PILImage
from io import BytesIO
from typing import Optional
import sys
import folder_paths
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import hashlib


def get_sha256(input: str):
    hash_object = hashlib.sha256()
    hash_object.update(input.encode("utf-8"))
    return hash_object.hexdigest()


def shuffle_arr_v2(arr, key):
    sha_key = get_sha256(key)
    arr_len = len(arr)
    s_idx = arr_len
    for i in range(arr_len):
        s_idx = arr_len - i - 1
        key_start = i % len(sha_key)
        key_slice = sha_key[key_start : key_start + 8]
        to_index = int(key_slice, 16) % (arr_len - i)
        arr[s_idx], arr[to_index] = arr[to_index], arr[s_idx]
    return arr


def shuffle_arr(arr, key):
    sha_key = get_sha256(key)
    key_len = len(sha_key)
    arr_len = len(arr)
    key_offset = 0
    for i in range(arr_len):
        to_index = int(get_sha256(sha_key[key_offset : key_offset + 8]), 16) % (
            arr_len - i
        )
        key_offset += 1
        if key_offset >= key_len:
            key_offset = 0
        arr[i], arr[to_index] = arr[to_index], arr[i]
    return arr


def encrypt_image(image: Image.Image, psw):
    width = image.width
    height = image.height
    x_arr = [i for i in range(width)]
    shuffle_arr(x_arr, psw)
    y_arr = [i for i in range(height)]
    shuffle_arr(y_arr, get_sha256(psw))
    pixels = image.load()
    for x in range(width):
        _x = x_arr[x]
        for y in range(height):
            _y = y_arr[y]
            pixels[x, y], pixels[_x, _y] = pixels[_x, _y], pixels[x, y]


def dencrypt_image(image: Image.Image, psw):
    width = image.width
    height = image.height
    x_arr = [i for i in range(width)]
    shuffle_arr(x_arr, psw)
    y_arr = [i for i in range(height)]
    shuffle_arr(y_arr, get_sha256(psw))
    pixels = image.load()
    for x in range(width - 1, -1, -1):
        _x = x_arr[x]
        for y in range(height - 1, -1, -1):
            _y = y_arr[y]
            pixels[x, y], pixels[_x, _y] = pixels[_x, _y], pixels[x, y]


def encrypt_image_v2(image: Image.Image, psw):
    width = image.width
    height = image.height
    x_arr = [i for i in range(width)]
    shuffle_arr(x_arr, psw)
    y_arr = [i for i in range(height)]
    shuffle_arr(y_arr, get_sha256(psw))
    pixel_array = np.array(image)

    for y in range(height):
        _y = y_arr[y]
        temp = pixel_array[y].copy()
        pixel_array[y] = pixel_array[_y]
        pixel_array[_y] = temp
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))
    for x in range(width):
        _x = x_arr[x]
        temp = pixel_array[x].copy()
        pixel_array[x] = pixel_array[_x]
        pixel_array[_x] = temp
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    image.paste(Image.fromarray(pixel_array))
    return image


def dencrypt_image_v2(image: Image.Image, psw):
    width = image.width
    height = image.height
    x_arr = [i for i in range(width)]
    shuffle_arr(x_arr, psw)
    y_arr = [i for i in range(height)]
    shuffle_arr(y_arr, get_sha256(psw))
    pixel_array = np.array(image)

    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))
    for x in range(width - 1, -1, -1):
        _x = x_arr[x]
        temp = pixel_array[x].copy()
        pixel_array[x] = pixel_array[_x]
        pixel_array[_x] = temp
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))
    for y in range(height - 1, -1, -1):
        _y = y_arr[y]
        temp = pixel_array[y].copy()
        pixel_array[y] = pixel_array[_y]
        pixel_array[_y] = temp

    image.paste(Image.fromarray(pixel_array))
    return image


def encrypt_image_v3(image: Image.Image, psw):
    width = image.width
    height = image.height
    x_arr = np.arange(width)
    shuffle_arr_v2(x_arr, psw)
    y_arr = np.arange(height)
    shuffle_arr_v2(y_arr, get_sha256(psw))
    pixel_array = np.array(image)

    _pixel_array = pixel_array.copy()
    for x in range(height):
        pixel_array[x] = _pixel_array[y_arr[x]]
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    _pixel_array = pixel_array.copy()
    for x in range(width):
        pixel_array[x] = _pixel_array[x_arr[x]]
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    image.paste(Image.fromarray(pixel_array))
    return image


def decrypt_image_v3(image: Image.Image, psw):
    if image.mode != "RGB":
        image = image.convert("RGB")
    width = image.width
    height = image.height
    x_arr = np.arange(width)
    shuffle_arr_v2(x_arr, psw)
    y_arr = np.arange(height)
    shuffle_arr_v2(y_arr, get_sha256(psw))
    pixel_array = np.array(image, dtype=np.uint8)

    _pixel_array = pixel_array.copy()
    for x in range(height):
        pixel_array[y_arr[x]] = _pixel_array[x]
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    _pixel_array = pixel_array.copy()
    for x in range(width):
        pixel_array[x_arr[x]] = _pixel_array[x]
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

    return Image.fromarray(pixel_array)


_password = "123qwe"

if PILImage.Image.__name__ != "EncryptedImage":
    super_open = PILImage.open

    class EncryptedImage(PILImage.Image):
        __name__ = "EncryptedImage"

        @staticmethod
        def from_image(image: PILImage.Image):
            image = image.copy()
            img = EncryptedImage()
            img.im = image.im
            img._mode = image.im.mode
            if image.im.mode:
                try:
                    img.mode = image.im.mode
                except Exception as e:
                    """"""
            img._size = image.size
            img.format = image.format
            if image.mode in ("P", "PA"):
                if image.palette:
                    img.palette = image.palette.copy()
                else:
                    img.palette = ImagePalette.ImagePalette()
            img.info = image.info.copy()
            return img

        def save(self, fp, format=None, **params):
            filename = ""
            if isinstance(fp, Path):
                filename = str(fp)
            elif _util.is_path(fp):
                filename = fp
            elif fp == sys.stdout:
                try:
                    fp = sys.stdout.buffer
                except AttributeError:
                    pass
            if not filename and hasattr(fp, "name") and _util.is_path(fp.name):
                filename = fp.name

            if not filename or not _password:
                super().save(fp, format=format, **params)
                return

            if "Encrypt" in self.info and (
                self.info["Encrypt"] == "pixel_shuffle"
                or self.info["Encrypt"] == "pixel_shuffle_2"
                or self.info["Encrypt"] == "pixel_shuffle_3"
            ):
                super().save(fp, format=format, **params)
                return

            encrypt_image_v3(self, get_sha256(_password))
            self.format = PngImagePlugin.PngImageFile.format
            if self.info:
                self.info["Encrypt"] = "pixel_shuffle_3"
            pnginfo = params.get("pnginfo", PngImagePlugin.PngInfo())
            if not pnginfo:
                pnginfo = PngImagePlugin.PngInfo()
                for key in (self.info or {}).keys():
                    if self.info[key]:
                        pnginfo.add_text(key, str(self.info[key]))
            pnginfo.add_text("Encrypt", "pixel_shuffle_3")
            pnginfo.add_text(
                "EncryptPwdSha", get_sha256(f"{get_sha256(_password)}Encrypt")
            )
            params.update(pnginfo=pnginfo)
            super().save(fp, format=self.format, **params)
            decrypt_image_v3(self, get_sha256(_password))  # Updated to decrypt_image_v3
            if self.info:
                self.info["Encrypt"] = None

    def open(fp, *args, **kwargs):
        image = super_open(fp, *args, **kwargs)
        if (
            _password
            and image.format.lower() == PngImagePlugin.PngImageFile.format.lower()
        ):
            pnginfo = image.info or {}
            if "Encrypt" in pnginfo and pnginfo["Encrypt"] == "pixel_shuffle":
                dencrypt_image(image, get_sha256(_password))
                pnginfo["Encrypt"] = None
                image = EncryptedImage.from_image(image=image)
                return image
            if "Encrypt" in pnginfo and pnginfo["Encrypt"] == "pixel_shuffle_2":
                dencrypt_image_v2(image, get_sha256(_password))
                pnginfo["Encrypt"] = None
                image = EncryptedImage.from_image(image=image)
                return image
            if "Encrypt" in pnginfo and pnginfo["Encrypt"] == "pixel_shuffle_3":
                decrypted_image = decrypt_image_v3(image, get_sha256(_password))
                pnginfo = image.info.copy()
                pnginfo["Encrypt"] = None
                image = EncryptedImage.from_image(image=decrypted_image)
                return image
        return EncryptedImage.from_image(image=image)

    PILImage.Image = EncryptedImage
    PILImage.open = open

    print("图片加密插件加载成功")


class EncryptImage:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "encryptd")
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "password": ("STRING", {"default": "123qwe"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "set_password"

    OUTPUT_NODE = True

    CATEGORY = "utils"

    def set_password(
        self,
        images,
        password,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        global _password
        _password = password
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {
                    "filename": file,
                    "subfolder": os.path.join("encryptd", subfolder),
                    "type": self.type,
                    "channel": "rgb",
                }
            )
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"EncryptImage": EncryptImage}
