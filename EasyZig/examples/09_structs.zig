// Chapter 9: Structs
// Custom data types with named fields

const std = @import("std");

// Basic struct definition
const Point = struct {
    x: i32,
    y: i32,
};

// Struct with default values
const Config = struct {
    width: u32 = 800,
    height: u32 = 600,
    fullscreen: bool = false,
    title: []const u8 = "My App",
};

// Struct with methods
const Rectangle = struct {
    width: f64,
    height: f64,

    // Method (takes self)
    pub fn area(self: Rectangle) f64 {
        return self.width * self.height;
    }

    // Method with pointer (can modify)
    pub fn scale(self: *Rectangle, factor: f64) void {
        self.width *= factor;
        self.height *= factor;
    }

    // Associated function (no self)
    pub fn square(size: f64) Rectangle {
        return Rectangle{ .width = size, .height = size };
    }
};

pub fn main() void {
    basic_structs();
    struct_methods();
    nested_structs();
    anonymous_structs();
}

fn basic_structs() void {
    std.debug.print("=== Basic Structs ===\n", .{});

    // Create instance with all fields
    const p1 = Point{ .x = 10, .y = 20 };
    std.debug.print("p1: ({d}, {d})\n", .{ p1.x, p1.y });

    // Create with default values
    const cfg1 = Config{};
    std.debug.print("Default config: {d}x{d}\n", .{ cfg1.width, cfg1.height });

    // Override some defaults
    const cfg2 = Config{
        .width = 1920,
        .height = 1080,
        .fullscreen = true,
    };
    std.debug.print("Custom config: {d}x{d}, fullscreen={}\n", .{
        cfg2.width,
        cfg2.height,
        cfg2.fullscreen,
    });

    // Mutable struct
    var p2 = Point{ .x = 0, .y = 0 };
    p2.x = 100;
    p2.y = 200;
    std.debug.print("p2: ({d}, {d})\n", .{ p2.x, p2.y });
}

fn struct_methods() void {
    std.debug.print("\n=== Struct Methods ===\n", .{});

    // Create rectangle
    var rect = Rectangle{ .width = 10.0, .height = 5.0 };
    std.debug.print("Rectangle: {d} x {d}\n", .{ rect.width, rect.height });
    std.debug.print("Area: {d}\n", .{rect.area()});

    // Modify with method
    rect.scale(2.0);
    std.debug.print("After scale(2): {d} x {d}\n", .{ rect.width, rect.height });
    std.debug.print("New area: {d}\n", .{rect.area()});

    // Associated function
    const sq = Rectangle.square(7.0);
    std.debug.print("Square: {d} x {d}, area={d}\n", .{ sq.width, sq.height, sq.area() });
}

// Nested struct
const Person = struct {
    name: []const u8,
    age: u32,
    address: Address,

    const Address = struct {
        street: []const u8,
        city: []const u8,
        zip: []const u8,
    };

    pub fn describe(self: Person) void {
        std.debug.print("{s}, age {d}\n", .{ self.name, self.age });
        std.debug.print("Lives at: {s}, {s} {s}\n", .{
            self.address.street,
            self.address.city,
            self.address.zip,
        });
    }
};

fn nested_structs() void {
    std.debug.print("\n=== Nested Structs ===\n", .{});

    const john = Person{
        .name = "John Doe",
        .age = 30,
        .address = .{
            .street = "123 Main St",
            .city = "Springfield",
            .zip = "12345",
        },
    };

    john.describe();
}

fn anonymous_structs() void {
    std.debug.print("\n=== Anonymous Structs ===\n", .{});

    // Anonymous struct literal
    const point = .{
        .x = @as(i32, 10),
        .y = @as(i32, 20),
    };
    std.debug.print("Anonymous point: ({d}, {d})\n", .{ point.x, point.y });

    // Tuple (anonymous struct with numbered fields)
    const tuple = .{ 42, "hello", 3.14 };
    std.debug.print("Tuple: {d}, {s}, {d}\n", .{ tuple[0], tuple[1], tuple[2] });
}

// Packed struct for memory layout control
const PackedData = packed struct {
    flag1: bool,
    flag2: bool,
    flag3: bool,
    value: u5, // 5 bits
};

pub fn packed_structs() void {
    std.debug.print("\n=== Packed Structs ===\n", .{});

    const data = PackedData{
        .flag1 = true,
        .flag2 = false,
        .flag3 = true,
        .value = 15,
    };

    std.debug.print("Size of PackedData: {d} bytes\n", .{@sizeOf(PackedData)});
    std.debug.print("flags: {}, {}, {}, value: {d}\n", .{
        data.flag1,
        data.flag2,
        data.flag3,
        data.value,
    });
}
