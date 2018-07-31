import resource
import objgraph


def dump_heap(#h,
              i):
    """
    From https://dzone.com/articles/python-memory-issues-tips-and-tricks
    Args:
        h: The heap (from hp = hpy(), h = hp.heap())
        i: Identifier str
    Returns:
        None
    """
    print "Dumping stats at: {0}".format(i)
    print "Memory usage: {0} (MB)".format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    print "Most common types:"
    objgraph.show_most_common_types()

    # print "heap is:"
    # print "{0}".format(h)
    #
    # by_refs = h.byrcs
    # print "by references: {0}".format(by_refs)
    #
    # print "More stats for top element.."
    # print "By clodo (class or dict owner): {0}".format(by_refs[0].byclodo)
    # print "By size: {0}".format(by_refs[0].bysize)
    # print "By id: {0}".format(by_refs[0].byid)
