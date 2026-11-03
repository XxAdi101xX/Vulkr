/* Copyright (c) 2018-2019, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdlib>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#define LOGGER_FORMAT "[%^%l%$] %v"
#define PROJECT_NAME "Vulkr"

#define LOGD(...) spdlog::debug(__VA_ARGS__);
#define LOGI(...) spdlog::info(__VA_ARGS__);
#define LOGW(...) spdlog::warn(__VA_ARGS__);
#define LOGE(...) spdlog::error(__VA_ARGS__);
#define LOGEANDABORT(...)                                                          \
	do                                                                             \
	{                                                                              \
		spdlog::error("[{}:{}] {}", __FILE__, __LINE__, fmt::format(__VA_ARGS__)); \
		spdlog::default_logger()->flush();                                         \
		std::abort();                                                              \
	} while (0);

/* Assertion that reports through the logger: unlike assert(), the failure is tagged like every other error and carries
   the file and line, and the sink is flushed before aborting so the message survives a crash. Libraries whose
   assertions can be redirected (VMA) point their own macros at VULKR_ASSERT_IMPL */
#define VULKR_ASSERT_IMPL(expr)                                                   \
	do                                                                            \
	{                                                                             \
		if (!(expr))                                                              \
		{                                                                         \
			spdlog::error("[{}:{}] Assertion failed: {}", __FILE__, __LINE__, #expr); \
			spdlog::default_logger()->flush();                                    \
			std::abort();                                                         \
		}                                                                         \
	} while (0);

#if defined(VULKR_DEBUG) || defined(_DEBUG)
	#define VULKR_ASSERT(expr) VULKR_ASSERT_IMPL(expr)
#else
	#define VULKR_ASSERT(expr) ((void)0)
#endif