/* Copyright (c) 2020 Adithya Venkatarao
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "image_view.h"

namespace vulkr
{

ImageView::ImageView(Image &image, VkImageViewType viewType, VkImageCreateFlags aspectMask, VkFormat format) :
	device{ image.getDevice() },
	image{ &image }
{
	subresourceRange.aspectMask = aspectMask;
	subresourceRange.baseMipLevel = 0;
	subresourceRange.levelCount = this->image->getSubresource().mipLevel;
	subresourceRange.baseArrayLayer = 0;
	subresourceRange.layerCount = this->image->getSubresource().arrayLayer;

	VkImageViewCreateInfo createInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
	createInfo.image = this->image->getHandle();
	createInfo.viewType = viewType;
	createInfo.format = format;
	createInfo.subresourceRange = subresourceRange;

	VK_CHECK(vkCreateImageView(device.getHandle(), &createInfo, nullptr, &handle));
}

ImageView::~ImageView()
{
	if (handle != VK_NULL_HANDLE)
	{
		vkDestroyImageView(device.getHandle(), handle, nullptr);
	}
}

const Image& ImageView::getImage() const
{
	return *image;
}

void ImageView::setImage(Image &image)
{
	this->image = &image;
}

VkImageView ImageView::getHandle() const
{
	return handle;
}

VkFormat ImageView::getFormat() const
{
	return image->getFormat();
}

VkImageSubresourceRange ImageView::getSubresourceRange() const
{
	return subresourceRange;
}

} // namespace vulkr