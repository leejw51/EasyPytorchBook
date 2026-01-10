// Chapter 15: Optionals
// Optional types for nullable values

const std = @import("std");

pub fn main() void {
    basic_optionals();
    optional_unwrapping();
    optional_pointers();
    optional_chaining();
}

fn basic_optionals() void {
    std.debug.print("=== Basic Optionals ===\n", .{});

    // Declare optional type
    var maybe_num: ?i32 = 42;
    std.debug.print("maybe_num = {?d}\n", .{maybe_num});

    // Set to null
    maybe_num = null;
    std.debug.print("maybe_num (null) = {?d}\n", .{maybe_num});

    // Optional with value
    const some: ?[]const u8 = "Hello";
    const none: ?[]const u8 = null;

    std.debug.print("some = {?s}\n", .{some});
    std.debug.print("none = {?s}\n", .{none});

    // Check if has value
    if (some != null) {
        std.debug.print("some is not null\n", .{});
    }
    if (none == null) {
        std.debug.print("none is null\n", .{});
    }
}

fn optional_unwrapping() void {
    std.debug.print("\n=== Optional Unwrapping ===\n", .{});

    const maybe_value: ?i32 = 42;
    const no_value: ?i32 = null;

    // If unwrap (payload capture)
    if (maybe_value) |value| {
        std.debug.print("Value is: {d}\n", .{value});
    } else {
        std.debug.print("No value\n", .{});
    }

    // Unwrap with else
    if (no_value) |value| {
        std.debug.print("Value is: {d}\n", .{value});
    } else {
        std.debug.print("no_value is null\n", .{});
    }

    // orelse - provide default value
    const val1 = maybe_value orelse 0;
    const val2 = no_value orelse -1;
    std.debug.print("val1 = {d}, val2 = {d}\n", .{ val1, val2 });

    // orelse with unreachable (assert not null)
    const must_exist: ?i32 = 100;
    const guaranteed = must_exist orelse unreachable;
    std.debug.print("guaranteed = {d}\n", .{guaranteed});

    // .? operator (shorthand for orelse unreachable)
    const direct = must_exist.?;
    std.debug.print("direct = {d}\n", .{direct});

    // While loop with optional
    var opt: ?i32 = 5;
    std.debug.print("Countdown: ", .{});
    while (opt) |*val| {
        std.debug.print("{d} ", .{val.*});
        if (val.* > 0) {
            val.* -= 1;
        } else {
            opt = null;
        }
    }
    std.debug.print("\n", .{});
}

// Function returning optional
fn find(haystack: []const u8, needle: u8) ?usize {
    for (haystack, 0..) |byte, index| {
        if (byte == needle) return index;
    }
    return null;
}

fn get_element(arr: []const i32, index: usize) ?i32 {
    if (index >= arr.len) return null;
    return arr[index];
}

fn optional_pointers() void {
    std.debug.print("\n=== Optional Pointers ===\n", .{});

    // Find in string
    const text = "Hello, World!";
    if (find(text, 'W')) |index| {
        std.debug.print("'W' found at index {d}\n", .{index});
    }

    if (find(text, 'Z')) |index| {
        std.debug.print("'Z' found at index {d}\n", .{index});
    } else {
        std.debug.print("'Z' not found\n", .{});
    }

    // Safe array access
    const numbers = [_]i32{ 10, 20, 30 };

    if (get_element(&numbers, 1)) |val| {
        std.debug.print("numbers[1] = {d}\n", .{val});
    }

    if (get_element(&numbers, 10)) |val| {
        std.debug.print("numbers[10] = {d}\n", .{val});
    } else {
        std.debug.print("Index 10 out of bounds\n", .{});
    }

    // Optional pointer
    var x: i32 = 42;
    var ptr: ?*i32 = &x;

    if (ptr) |p| {
        std.debug.print("Pointed value: {d}\n", .{p.*});
        p.* = 100;
    }

    std.debug.print("x after modification: {d}\n", .{x});

    ptr = null;
    if (ptr) |p| {
        std.debug.print("Value: {d}\n", .{p.*});
    } else {
        std.debug.print("Pointer is null\n", .{});
    }
}

// Linked list node using optional
const Node = struct {
    value: i32,
    next: ?*Node,

    fn traverse(self: *Node) void {
        var current: ?*Node = self;
        std.debug.print("List: ", .{});
        while (current) |node| {
            std.debug.print("{d} -> ", .{node.value});
            current = node.next;
        }
        std.debug.print("null\n", .{});
    }
};

fn optional_chaining() void {
    std.debug.print("\n=== Optional Chaining ===\n", .{});

    // Create linked list
    var node3 = Node{ .value = 30, .next = null };
    var node2 = Node{ .value = 20, .next = &node3 };
    var node1 = Node{ .value = 10, .next = &node2 };

    node1.traverse();

    // Access through optional chain
    if (node1.next) |n2| {
        if (n2.next) |n3| {
            std.debug.print("Third node value: {d}\n", .{n3.value});
        }
    }
}
