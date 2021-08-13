/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

namespace apex {

    class cuda_thread_node {
    public:
        uint32_t _device;
        uint32_t _context;
        uint32_t _stream;
        apex_async_activity_t _activity;
        cuda_thread_node(uint32_t device, uint32_t context, uint32_t stream,
            apex_async_activity_t activity) :
            _device(device), _context(context), _stream(stream),
            _activity(activity) { }
        bool operator==(const cuda_thread_node &rhs) const {
            return (_device   == rhs._device &&
                    _context  == rhs._context &&
                    _stream   == rhs._stream &&
                    _activity == rhs._activity);
        }
        bool operator<(const cuda_thread_node &rhs) const {
            if (_device<rhs._device) {
                return true;
            } else if (_device == rhs._device && _context < rhs._context) {
                return true;
            } else if (_device == rhs._device && _context == rhs._context &&
                _stream < rhs._stream) {
                return true;
            } else if (_device == rhs._device && _context == rhs._context &&
                _stream == rhs._stream && _activity < rhs._activity &&
                apex_options::use_otf2()) {
                return true;
            }
            return false;
        }
    };

    class hip_thread_node {
    public:
        uint32_t _device;
        uint32_t _queue;
        apex_async_activity_t _activity;
        hip_thread_node(uint32_t device, uint32_t command_queue,
            apex_async_activity_t activity) :
            _device(device), _queue(command_queue),
            _activity(activity) { }
        bool operator==(const hip_thread_node &rhs) const {
            return (_device   == rhs._device &&
                    _queue    == rhs._queue &&
                    _activity == rhs._activity);
        }
        bool operator<(const hip_thread_node &rhs) const {
            if (_device<rhs._device) {
                return true;
            } else if (_device == rhs._device && _queue < rhs._queue) {
                return true;
            } else if (_device == rhs._device && _queue == rhs._queue &&
                _activity < rhs._activity && apex_options::use_otf2()) {
                return true;
            }
            return false;
        }
    };

    class dummy_thread_node {
    public:
        uint32_t _device;
        apex_async_activity_t _activity;
        dummy_thread_node(uint32_t device, apex_async_activity_t activity) :
            _device(device), _activity(activity) { }
        bool operator==(const dummy_thread_node &rhs) const {
            return (_device   == rhs._device && _activity == rhs._activity);
        }
        bool operator<(const dummy_thread_node &rhs) const {
            if (_device<rhs._device) {
                return true;
            } else if (_device == rhs._device &&
                _activity < rhs._activity && apex_options::use_otf2()) {
                return true;
            }
            return false;
        }
    };

}

#ifdef APEX_WITH_CUDA
using async_thread_node = apex::cuda_thread_node;
#endif

#ifdef APEX_WITH_HIP
using async_thread_node = apex::hip_thread_node;
#endif

#if !defined(APEX_WITH_CUDA) && !defined(APEX_WITH_HIP)
using async_thread_node = apex::dummy_thread_node;
#endif
