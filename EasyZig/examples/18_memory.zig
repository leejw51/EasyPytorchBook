// Chapter 18: Memory Management
// Allocators and memory in Zig

const std = @import("std");

pub fn main() !void {
    stack_memory();
    try heap_allocation();
    try arraylist_example();
    try arena_example();
}

fn stack_memory() void {
    std.debug.print("=== Stack Memory ===\n", .{});

    // Stack-allocated variables
    var x: i32 = 42;
    var arr: [10]i32 = undefined;

    for (&arr, 0..) |*item, i| {
        item.* = @as(i32, @intCast(i * 2));
    }

    std.debug.print("x = {d}\n", .{x});
    std.debug.print("arr = {any}\n", .{arr});

    // Stack buffer
    var buffer: [256]u8 = undefined;
    const message = "Hello, Stack!";
    @memcpy(buffer[0..message.len], message);
    std.debug.print("buffer: {s}\n", .{buffer[0..message.len]});
}

fn heap_allocation() !void {
    std.debug.print("\n=== Heap Allocation ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Allocate single value
    const ptr = try allocator.create(i32);
    defer allocator.destroy(ptr);
    ptr.* = 100;
    std.debug.print("Allocated value: {d}\n", .{ptr.*});

    // Allocate slice
    const slice = try allocator.alloc(i32, 5);
    defer allocator.free(slice);

    for (slice, 0..) |*item, i| {
        item.* = @as(i32, @intCast((i + 1) * 10));
    }
    std.debug.print("Allocated slice: {any}\n", .{slice});
}

fn arraylist_example() !void {
    std.debug.print("\n=== ArrayList ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Dynamic array
    var list = std.ArrayList(i32).init(allocator);
    defer list.deinit();

    try list.append(10);
    try list.append(20);
    try list.append(30);
    try list.appendSlice(&[_]i32{ 40, 50 });

    std.debug.print("ArrayList: {any}\n", .{list.items});
    std.debug.print("Length: {d}, Capacity: {d}\n", .{ list.items.len, list.capacity });

    _ = list.pop();
    std.debug.print("After pop: {any}\n", .{list.items});
}

fn arena_example() !void {
    std.debug.print("\n=== Arena Allocator ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Arena for bulk allocations
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit(); // Frees all at once!

    const allocator = arena.allocator();

    // Make many allocations
    var ptrs: [10]*i32 = undefined;
    for (&ptrs, 0..) |*ptr, i| {
        ptr.* = try allocator.create(i32);
        ptr.*.* = @as(i32, @intCast(i * i));
    }

    std.debug.print("Arena values: ", .{});
    for (ptrs) |ptr| {
        std.debug.print("{d} ", .{ptr.*});
    }
    std.debug.print("\n", .{});

    // Allocate strings
    const s1 = try std.fmt.allocPrint(allocator, "Item {d}", .{1});
    const s2 = try std.fmt.allocPrint(allocator, "Item {d}", .{2});
    std.debug.print("{s}, {s}\n", .{ s1, s2 });

    // No need to free individually - arena.deinit() frees everything
}

// HashMap example
pub fn hashmap_example() !void {
    std.debug.print("\n=== HashMap ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var map = std.StringHashMap(i32).init(allocator);
    defer map.deinit();

    try map.put("one", 1);
    try map.put("two", 2);
    try map.put("three", 3);

    if (map.get("two")) |value| {
        std.debug.print("map['two'] = {d}\n", .{value});
    }

    // Iterate
    var iter = map.iterator();
    while (iter.next()) |entry| {
        std.debug.print("{s} = {d}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
}
