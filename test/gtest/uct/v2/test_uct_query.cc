/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
}

#include <gtest/uct/uct_p2p_test.h>

class test_uct_query : public uct_p2p_test {
public:
    test_uct_query() : uct_p2p_test(0)
    {
    }
};

UCS_TEST_P(test_uct_query, query_perf)
{
    uct_iface_perf_attr_t perf_attr;
    ucs_status_t status;    

    perf_attr.field_mask         = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH |
                                   UCT_PERF_ATTR_FIELD_LATENCY;
    perf_attr.operation          = UCT_OP_AM_SHORT;
    perf_attr.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    perf_attr.remote_memory_type = UCS_MEMORY_TYPE_HOST;
    status                       = uct_iface_estimate_perf(sender().iface(),
                                                           &perf_attr);
    EXPECT_EQ(status, UCS_OK);

    perf_attr.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    perf_attr.operation          = UCT_OP_PUT_SHORT;
    status                       = uct_iface_estimate_perf(sender().iface(),
                                                           &perf_attr);

    /* At least one type of bandwidth must be non-zero */
    EXPECT_NE(0, perf_attr.bandwidth.shared + perf_attr.bandwidth.dedicated);

    if (has_transport("self")) {
        /* The latency in "self" is 0 */
        EXPECT_EQ(perf_attr.latency.c, 0);
    } else {
        EXPECT_NE(perf_attr.latency.c, 0);
    }

    if (has_transport("cuda_copy") || has_transport("gdr_copy")) {
        uct_iface_perf_attr_t perf_attr_get;
        perf_attr_get.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH |
                                   UCT_PERF_ATTR_FIELD_LATENCY;
        perf_attr_get.operation  = UCT_OP_GET_SHORT;
        status = uct_iface_estimate_perf(sender().iface(), &perf_attr_get);
        EXPECT_EQ(status, UCS_OK);

        /* Put and get operations have different bandwidth in cuda_copy
           and gdr_copy transports */
        EXPECT_NE(perf_attr.latency.c, 0);

    }
}

UCT_INSTANTIATE_TEST_CASE(test_uct_query)
