# 쉬운 Zig

*Zig 프로그래밍 언어에 대한 간단하고 실용적인 가이드*

---

## 목차

- [0. Hello World](#0-hello-world)
- [1. Zig 소개](#1-zig-소개)
- [2. 변수와 타입](#2-변수와-타입)
- [3. 제어 흐름](#3-제어-흐름)
- [4. 함수](#4-함수)
- [5. 배열과 슬라이스](#5-배열과-슬라이스)
- [6. 구조체](#6-구조체)
- [7. 열거형과 유니온](#7-열거형과-유니온)
- [8. 에러 처리](#8-에러-처리)
- [9. 옵셔널](#9-옵셔널)
- [10. 포인터](#10-포인터)
- [11. 메모리 할당](#11-메모리-할당)
- [12. 컴파일 타임 실행](#12-컴파일-타임-실행)
- [13. 테스팅](#13-테스팅)
- [14. 빌드 시스템](#14-빌드-시스템)

---

## 0. Hello World


가장 간단한 Zig 프로그램으로 시작한 다음 메모리 할당을 살펴보겠습니다.

## 실행 방법

코드를 파일(예: `hello.zig`)에 저장하고 다음 명령으로 실행합니다:

```bash
zig run hello.zig
```

이게 전부입니다! Zig는 한 번의 명령으로 프로그램을 컴파일하고 실행합니다.

## 다른 빌드 방법

```
┌─────────────────────────────────────────┐
│  zig run file.zig      # 컴파일 후 실행  │
│  zig build-exe file.zig # 컴파일만       │
│  ./file                 # 바이너리 실행   │
└─────────────────────────────────────────┘
```


```
Zig 프로그램 실행:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  hello.zig  │ -> │  zig run    │ -> │    출력     │
│   (소스)    │    │  (컴파일)   │    │   (결과)    │
└─────────────┘    └─────────────┘    └─────────────┘

defer를 사용한 메모리 관리:

    ┌────────────────────────────────┐
    │  const x = try allocate();     │
    │  defer free(x);  <- 등록됨     │
    │                                │
    │  ... x 사용 ...                │
    │                                │
    └────────────────────────────────┘
              │
              ▼ (스코프 종료)
    ┌────────────────────────────────┐
    │  free(x) <- 실행됨!            │
    └────────────────────────────────┘
```


### 예제: hello_world

가장 간단한 Zig 프로그램 - Hello World를 콘솔에 출력

```zig
const std = @import("std");

pub fn main() void {
    std.debug.print("Hello, World!\n", .{});
}
```

**출력:**
```
Hello, World!
```


### 예제: allocator_example

defer를 사용한 자동 정리와 메모리 할당의 종합적인 예제

```zig
const std = @import("std");

// Function that takes an allocator and creates a dynamic string
fn createGreeting(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    // Allocate memory for the greeting
    const greeting = try std.fmt.allocPrint(allocator, "Hello, {s}! Welcome to Zig.", .{name});
    return greeting;
}

// Function that creates a dynamic array
fn createNumbers(allocator: std.mem.Allocator, count: usize) ![]u32 {
    // Allocate an array of u32
    const numbers = try allocator.alloc(u32, count);

    // Fill with values
    for (numbers, 0..) |*num, i| {
        num.* = @intCast(i * 10);
    }

    return numbers;
}

pub fn main() !void {
    // Create a General Purpose Allocator (GPA)
    // GPA detects memory leaks and double-frees in debug mode
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("Memory leak detected!\n", .{});
        }
    }

    const allocator = gpa.allocator();

    // Example 1: Using defer for automatic cleanup
    std.debug.print("=== Example 1: String with defer ===\n", .{});
    {
        const greeting = try createGreeting(allocator, "Alice");
        defer allocator.free(greeting); // Automatically freed when scope ends

        std.debug.print("{s}\n", .{greeting});
    } // greeting is freed here

    // Example 2: Dynamic array with defer
    std.debug.print("\n=== Example 2: Array with defer ===\n", .{});
    {
        const numbers = try createNumbers(allocator, 5);
        defer allocator.free(numbers); // Automatically freed when scope ends

        std.debug.print("Numbers: ", .{});
        for (numbers) |n| {
            std.debug.print("{d} ", .{n});
        }
        std.debug.print("\n", .{});
    } // numbers is freed here

    // Example 3: Multiple allocations with defer (LIFO order)
    std.debug.print("\n=== Example 3: Multiple defers (LIFO order) ===\n", .{});
    {
        const str1 = try createGreeting(allocator, "Bob");
        defer {
            std.debug.print("Freeing str1\n", .{});
            allocator.free(str1);
        }

        const str2 = try createGreeting(allocator, "Charlie");
        defer {
            std.debug.print("Freeing str2\n", .{});
            allocator.free(str2);
        }

        std.debug.print("{s}\n", .{str1});
        std.debug.print("{s}\n", .{str2});
    } // str2 freed first, then str1 (LIFO)

    // Example 4: Using ArrayListUnmanaged with allocator
    std.debug.print("\n=== Example 4: ArrayListUnmanaged ===\n", .{});
    {
        var list: std.ArrayListUnmanaged(i32) = .{};
        defer list.deinit(allocator); // Pass allocator to deinit

        try list.append(allocator, 100);
        try list.append(allocator, 200);
        try list.append(allocator, 300);

        std.debug.print("ArrayList items: ", .{});
        for (list.items) |item| {
            std.debug.print("{d} ", .{item});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\n=== All memory properly freed! ===\n", .{});
}
```

**출력:**
```
=== Example 1: String with defer ===
Hello, Alice! Welcome to Zig.

=== Example 2: Array with defer ===
Numbers: 0 10 20 30 40 

=== Example 3: Multiple defers (LIFO order) ===
Hello, Bob! Welcome to Zig.
Hello, Charlie! Welcome to Zig.
Freeing str2
Freeing str1

=== Example 4: ArrayListUnmanaged ===
ArrayList items: 100 200 300 

=== All memory properly freed! ===
```


---

## 1. Zig 소개


Zig는 다음을 위해 설계된 현대적인 시스템 프로그래밍 언어입니다:
- **성능**: 숨겨진 제어 흐름이나 할당 없음
- **안전성**: 선택적 안전 검사, 정의되지 않은 동작 없음
- **단순성**: 숨겨진 마법 없음, 읽기 쉬운 코드
- **상호운용성**: C ABI와 직접 호환

## 시작하기

https://ziglang.org/download/ 에서 Zig를 설치하세요.

```
┌─────────────────────────────────────────┐
│           Zig 작업 흐름                  │
├─────────────────────────────────────────┤
│  1. 코드 작성  →  main.zig              │
│  2. 컴파일     →  zig build-exe         │
│  3. 실행       →  ./main                │
│                                         │
│  또는 간단히:  →  zig run main.zig      │
└─────────────────────────────────────────┘
```


```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    소스      │ -> │   컴파일러   │ -> │   바이너리   │
│  main.zig    │    │     zig      │    │   ./main     │
└──────────────┘    └──────────────┘    └──────────────┘
```


### 예제: hello_world

고전적인 Hello World 프로그램

```zig
const std = @import("std");

// This is your first Zig program!
// std.debug.print outputs to stderr
pub fn main() void {
    std.debug.print("Hello, Zig!\n", .{});
}
```

**출력:**
```
Hello, Zig!
```


---

## 2. 변수와 타입


Zig는 명시적 타입을 가진 간단한 타입 시스템을 가지고 있습니다.

## 변수 선언

```
┌─────────────────────────────────────────┐
│  const x = 5;    // 불변 (const)         │
│  var y = 10;     // 가변 (var)           │
│  var z: i32 = 0; // 명시적 타입          │
└─────────────────────────────────────────┘
```

## 기본 타입

| 타입     | 설명              | 크기    |
|----------|-------------------|---------|
| i8-i128  | 부호 있는 정수    | 1-16 B  |
| u8-u128  | 부호 없는 정수    | 1-16 B  |
| f32,f64  | 부동소수점        | 4,8 B   |
| bool     | 불리언            | 1 B     |
| void     | 값 없음           | 0 B     |


```
메모리 레이아웃 (리틀 엔디안):

u8:   [xxxxxxxx]           (1 바이트)
u16:  [xxxxxxxx][xxxxxxxx] (2 바이트)
u32:  [xxxx][xxxx][xxxx][xxxx] (4 바이트)

i8 범위:  -128 ~ 127
u8 범위:  0 ~ 255
```


### 예제: variables

변수 선언과 기본 타입

```zig
const std = @import("std");

pub fn main() void {
    // Immutable constant
    const message = "Hello";

    // Mutable variable
    var count: i32 = 0;
    count += 1;

    // Type inference
    const pi = 3.14159;

    // Explicit type
    const byte: u8 = 255;

    std.debug.print("message: {s}\n", .{message});
    std.debug.print("count: {}\n", .{count});
    std.debug.print("pi: {d:.2}\n", .{pi});
    std.debug.print("byte: {}\n", .{byte});
}
```

**출력:**
```
message: Hello
count: 1
pi: 3.14
byte: 255
```


### 예제: type_coercion

타입 변환과 강제 변환

```zig
const std = @import("std");

pub fn main() void {
    // Integer types
    const small: u8 = 10;
    const big: u32 = small; // Safe widening

    // Explicit cast for narrowing
    const large: u32 = 1000;
    const truncated: u8 = @truncate(large);

    std.debug.print("small: {}, big: {}\n", .{ small, big });
    std.debug.print("large: {}, truncated: {}\n", .{ large, truncated });
}
```

**출력:**
```
small: 10, big: 10
large: 1000, truncated: 232
```


---

## 3. 제어 흐름


Zig는 몇 가지 고유한 기능을 가진 표준 제어 흐름을 제공합니다.

## If/Else

```
┌─────────────────────────────────────────┐
│  if (조건) {                            │
│      // 참인 경우                       │
│  } else {                               │
│      // 거짓인 경우                     │
│  }                                      │
└─────────────────────────────────────────┘
```

## While 루프

```
┌─────────────────────────────────────────┐
│  while (조건) : (계속_표현식) {          │
│      // 루프 본문                       │
│  }                                      │
└─────────────────────────────────────────┘
```


```
제어 흐름 다이어그램:

    ┌───────┐
    │ 시작  │
    └───┬───┘
        │
    ┌───▼───┐     예     ┌───────┐
    │ 조건? ├────────────► 블록  │
    └───┬───┘            └───────┘
        │ 아니오
    ┌───▼───┐
    │  종료  │
    └───────┘
```


### 예제: if_else

If/else 문과 표현식

```zig
const std = @import("std");

pub fn main() void {
    const x: i32 = 42;

    // Basic if/else
    if (x > 0) {
        std.debug.print("x is positive\n", .{});
    } else if (x < 0) {
        std.debug.print("x is negative\n", .{});
    } else {
        std.debug.print("x is zero\n", .{});
    }

    // If as expression
    const abs_x = if (x < 0) -x else x;
    std.debug.print("abs(x) = {}\n", .{abs_x});
}
```

**출력:**
```
x is positive
abs(x) = 42
```


### 예제: loops

While과 for 루프

```zig
const std = @import("std");

pub fn main() void {
    // While loop
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        std.debug.print("while i = {}\n", .{i});
    }

    // For loop over range
    for (0..3) |j| {
        std.debug.print("for j = {}\n", .{j});
    }

    // For loop over array
    const arr = [_]i32{ 10, 20, 30 };
    for (arr) |val| {
        std.debug.print("val = {}\n", .{val});
    }
}
```

**출력:**
```
while i = 0
while i = 1
while i = 2
for j = 0
for j = 1
for j = 2
val = 10
val = 20
val = 30
```


### 예제: switch

Switch 표현식

```zig
const std = @import("std");

pub fn main() void {
    const day: u8 = 3;

    // Switch statement
    const name = switch (day) {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6, 7 => "Weekend",
        else => "Invalid",
    };

    std.debug.print("Day {}: {s}\n", .{ day, name });
}
```

**출력:**
```
Day 3: Wednesday
```


---

## 4. 함수


함수는 Zig에서 일급 값입니다.

## 함수 문법

```
┌─────────────────────────────────────────┐
│  fn 이름(매개변수: 타입) 반환타입 {       │
│      return 값;                         │
│  }                                      │
└─────────────────────────────────────────┘
```

## 특별한 기능
- 함수를 값으로 전달 가능
- 컴파일 타임 함수 평가
- `inline`을 사용한 인라인 함수


```
함수 호출 스택:

┌─────────────────────┐
│   main()            │  <- 스택 프레임
├─────────────────────┤
│   add(3, 4)         │  <- 새 프레임
│   a = 3, b = 4      │
│   return 7          │
├─────────────────────┤
│   (main으로 반환)    │
└─────────────────────┘
```


### 예제: basic_functions

기본 함수 정의

```zig
const std = @import("std");

// Simple function
fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Function with multiple return (tuple)
fn divmod(a: i32, b: i32) struct { quot: i32, rem: i32 } {
    return .{ .quot = @divTrunc(a, b), .rem = @mod(a, b) };
}

pub fn main() void {
    const sum = add(3, 4);
    std.debug.print("3 + 4 = {}\n", .{sum});

    const result = divmod(17, 5);
    std.debug.print("17 / 5 = {} remainder {}\n", .{ result.quot, result.rem });
}
```

**출력:**
```
3 + 4 = 7
17 / 5 = 3 remainder 2
```


### 예제: function_pointers

함수 포인터와 고차 함수

```zig
const std = @import("std");

fn double(x: i32) i32 {
    return x * 2;
}

fn triple(x: i32) i32 {
    return x * 3;
}

// Function that takes a function pointer
fn apply(value: i32, func: *const fn (i32) i32) i32 {
    return func(value);
}

pub fn main() void {
    const x: i32 = 5;

    std.debug.print("double({}) = {}\n", .{ x, apply(x, &double) });
    std.debug.print("triple({}) = {}\n", .{ x, apply(x, &triple) });
}
```

**출력:**
```
double(5) = 10
triple(5) = 15
```


---

## 5. 배열과 슬라이스


배열은 고정 크기를 가지고, 슬라이스는 배열의 뷰입니다.

## 배열 vs 슬라이스

```
┌─────────────────────────────────────────┐
│  배열: [N]T   - 컴파일 타임 고정 크기    │
│  슬라이스: []T - 런타임 배열 뷰          │
└─────────────────────────────────────────┘
```


```
메모리 내 배열:
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │  arr[5]i32
└───┴───┴───┴───┴───┘
  0   1   2   3   4

슬라이스 (뷰):
        ┌───────────┐
        │ ptr + len │
        └─────┬─────┘
              │
┌───┬───┬───┬─▼─┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │
└───┴───┴───┴───┴───┘
        └─────┘
        slice [1..4]
```


### 예제: arrays

배열 기초

```zig
const std = @import("std");

pub fn main() void {
    // Array with explicit size
    const arr1: [5]i32 = [_]i32{ 1, 2, 3, 4, 5 };

    // Array with inferred size
    const arr2 = [_]i32{ 10, 20, 30 };

    // Access elements
    std.debug.print("arr1[0] = {}\n", .{arr1[0]});
    std.debug.print("arr2.len = {}\n", .{arr2.len});

    // Iterate
    var sum: i32 = 0;
    for (arr1) |x| {
        sum += x;
    }
    std.debug.print("sum of arr1 = {}\n", .{sum});
}
```

**출력:**
```
arr1[0] = 1
arr2.len = 3
sum of arr1 = 15
```


### 예제: slices

슬라이스 - 배열의 뷰

```zig
const std = @import("std");

pub fn main() void {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };

    // Create slice from array
    const slice: []const i32 = arr[1..4]; // [2, 3, 4]

    std.debug.print("slice.len = {}\n", .{slice.len});

    for (slice, 0..) |val, i| {
        std.debug.print("slice[{}] = {}\n", .{ i, val });
    }
}
```

**출력:**
```
slice.len = 3
slice[0] = 2
slice[1] = 3
slice[2] = 4
```


### 예제: string_literals

바이트 슬라이스로서의 문자열

```zig
const std = @import("std");

pub fn main() void {
    // String literal is []const u8
    const hello = "Hello, World!";

    std.debug.print("String: {s}\n", .{hello});
    std.debug.print("Length: {}\n", .{hello.len});
    std.debug.print("First char: {c}\n", .{hello[0]});

    // Iterate over characters
    for (hello[0..5]) |c| {
        std.debug.print("{c} ", .{c});
    }
    std.debug.print("\n", .{});
}
```

**출력:**
```
String: Hello, World!
Length: 13
First char: H
H e l l o
```


---

## 6. 구조체


구조체는 관련 데이터를 함께 그룹화합니다.

## 구조체 정의

```
┌─────────────────────────────────────────┐
│  const Point = struct {                 │
│      x: i32,                            │
│      y: i32,                            │
│                                         │
│      fn init(x: i32, y: i32) Point {    │
│          return .{ .x = x, .y = y };    │
│      }                                  │
│  };                                     │
└─────────────────────────────────────────┘
```


```
구조체 메모리 레이아웃:

Point { x: i32, y: i32 }

┌─────────────┬─────────────┐
│    x: i32   │    y: i32   │
│   4 바이트  │   4 바이트  │
└─────────────┴─────────────┘
     총: 8 바이트
```


### 예제: basic_struct

메서드가 있는 구조체 정의

```zig
const std = @import("std");

// Define a struct
const Point = struct {
    x: i32,
    y: i32,

    // Method
    fn distance_from_origin(self: Point) f32 {
        const fx: f32 = @floatFromInt(self.x);
        const fy: f32 = @floatFromInt(self.y);
        return @sqrt(fx * fx + fy * fy);
    }
};

pub fn main() void {
    // Create instance
    const p = Point{ .x = 3, .y = 4 };

    std.debug.print("Point: ({}, {})\n", .{ p.x, p.y });
    std.debug.print("Distance: {d:.2}\n", .{p.distance_from_origin()});
}
```

**출력:**
```
Point: (3, 4)
Distance: 5.00
```


### 예제: struct_defaults

기본 필드 값이 있는 구조체

```zig
const std = @import("std");

const Config = struct {
    // Fields with default values
    width: u32 = 800,
    height: u32 = 600,
    title: []const u8 = "Untitled",
};

pub fn main() void {
    // Use all defaults
    const c1 = Config{};

    // Override some defaults
    const c2 = Config{ .width = 1920, .height = 1080 };

    std.debug.print("c1: {}x{} '{s}'\n", .{ c1.width, c1.height, c1.title });
    std.debug.print("c2: {}x{} '{s}'\n", .{ c2.width, c2.height, c2.title });
}
```

**출력:**
```
c1: 800x600 'Untitled'
c2: 1920x1080 'Untitled'
```


---

## 7. 열거형과 유니온


열거형은 명명된 값의 집합을 정의합니다. 태그드 유니온은 열거형과 데이터를 결합합니다.

## 열거형

```
┌─────────────────────────────────────────┐
│  const Color = enum { red, green, blue };│
└─────────────────────────────────────────┘
```

## 태그드 유니온

```
┌─────────────────────────────────────────┐
│  const Value = union(enum) {            │
│      int: i32,                          │
│      float: f64,                        │
│      none,                              │
│  };                                     │
└─────────────────────────────────────────┘
```


```
태그드 유니온 메모리:

Value union(enum):
┌─────────┬─────────────────┐
│   태그  │      데이터      │
│ (enum)  │ (가장 큰 필드)   │
└─────────┴─────────────────┘

태그 값: .int=0, .float=1, .boolean=2, .none=3
```


### 예제: enums

메서드가 있는 열거형

```zig
const std = @import("std");

const Direction = enum {
    north,
    south,
    east,
    west,

    // Enum method
    fn opposite(self: Direction) Direction {
        return switch (self) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
        };
    }
};

pub fn main() void {
    const dir = Direction.north;
    const opp = dir.opposite();

    std.debug.print("Direction: {}\n", .{dir});
    std.debug.print("Opposite: {}\n", .{opp});
}
```

**출력:**
```
Direction: .north
Opposite: .south
```


### 예제: tagged_union

타입 안전한 변형을 위한 태그드 유니온

```zig
const std = @import("std");

const Value = union(enum) {
    int: i32,
    float: f64,
    boolean: bool,
    none: void,
};

fn printValue(val: Value) void {
    switch (val) {
        .int => |i| std.debug.print("int: {}\n", .{i}),
        .float => |f| std.debug.print("float: {d:.2}\n", .{f}),
        .boolean => |b| std.debug.print("bool: {}\n", .{b}),
        .none => std.debug.print("none\n", .{}),
    }
}

pub fn main() void {
    const v1 = Value{ .int = 42 };
    const v2 = Value{ .float = 3.14 };
    const v3 = Value{ .boolean = true };
    const v4 = Value{ .none = {} };

    printValue(v1);
    printValue(v2);
    printValue(v3);
    printValue(v4);
}
```

**출력:**
```
int: 42
float: 3.14
bool: true
none
```


---

## 8. 에러 처리


Zig는 에러 유니온을 사용한 명시적 에러 처리를 사용합니다.

## 에러 유니온 타입

```
┌─────────────────────────────────────────┐
│  fn divide(a: i32, b: i32) !i32 {       │
│      if (b == 0) return error.DivByZero;│
│      return @divTrunc(a, b);            │
│  }                                      │
└─────────────────────────────────────────┘
```

## 에러 처리 옵션
- `try`: 에러를 상위로 전파
- `catch`: 에러 처리
- `orelse`: 에러 시 기본값


```
에러 유니온 타입:

    anyerror!T 또는 ErrorSet!T
         │
    ┌────┴────┐
    │         │
  에러       값
    │         │
┌───▼───┐ ┌───▼───┐
│ catch │ │ 성공  │
└───────┘ └───────┘
```


### 예제: error_basics

catch를 사용한 기본 에러 처리

```zig
const std = @import("std");

const MathError = error{
    DivisionByZero,
    Overflow,
};

fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return MathError.DivisionByZero;
    return @divTrunc(a, b);
}

pub fn main() void {
    // Using catch to handle error
    const result1 = divide(10, 2) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return;
    };
    std.debug.print("10 / 2 = {}\n", .{result1});

    // Using catch with default
    const result2 = divide(10, 0) catch 0;
    std.debug.print("10 / 0 = {} (default)\n", .{result2});
}
```

**출력:**
```
10 / 2 = 5
10 / 0 = 0 (default)
```


### 예제: try_keyword

try를 사용한 에러 전파

```zig
const std = @import("std");

const ParseError = error{InvalidChar};

fn parseDigit(c: u8) ParseError!u8 {
    if (c >= '0' and c <= '9') {
        return c - '0';
    }
    return ParseError.InvalidChar;
}

fn parseNumber(s: []const u8) ParseError!u32 {
    var result: u32 = 0;
    for (s) |c| {
        // try propagates error if parseDigit fails
        const digit = try parseDigit(c);
        result = result * 10 + digit;
    }
    return result;
}

pub fn main() void {
    if (parseNumber("123")) |num| {
        std.debug.print("Parsed: {}\n", .{num});
    } else |err| {
        std.debug.print("Parse error: {}\n", .{err});
    }

    if (parseNumber("12x")) |num| {
        std.debug.print("Parsed: {}\n", .{num});
    } else |err| {
        std.debug.print("Parse error: {}\n", .{err});
    }
}
```

**출력:**
```
Parsed: 123
Parse error: error.InvalidChar
```


---

## 9. 옵셔널


옵셔널은 존재할 수도 있고 없을 수도 있는 값을 나타냅니다.

## 옵셔널 타입

```
┌─────────────────────────────────────────┐
│  var x: ?i32 = null;   // 값 없음       │
│  x = 42;               // 값 있음       │
│                                         │
│  if (x) |val| {        // 언래핑        │
│      // val 사용                        │
│  }                                      │
└─────────────────────────────────────────┘
```


```
옵셔널 타입 ?T:

┌─────────────┐
│   ?i32      │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
 null     값
   │       │
┌──▼──┐ ┌──▼──┐
│비어있│ │ 42  │
│  음  │ │     │
└─────┘ └─────┘
```


### 예제: optionals

옵셔널 타입과 언래핑

```zig
const std = @import("std");

fn find_first_even(arr: []const i32) ?i32 {
    for (arr) |x| {
        if (@mod(x, 2) == 0) return x;
    }
    return null; // Not found
}

pub fn main() void {
    const arr1 = [_]i32{ 1, 3, 4, 7, 9 };
    const arr2 = [_]i32{ 1, 3, 5, 7, 9 };

    // Using if to unwrap
    if (find_first_even(&arr1)) |val| {
        std.debug.print("First even: {}\n", .{val});
    } else {
        std.debug.print("No even number found\n", .{});
    }

    // Using orelse for default
    const result = find_first_even(&arr2) orelse -1;
    std.debug.print("Result with default: {}\n", .{result});
}
```

**출력:**
```
First even: 4
Result with default: -1
```


### 예제: optional_pointers

연결 구조를 위한 옵셔널 포인터

```zig
const std = @import("std");

const Node = struct {
    value: i32,
    next: ?*Node, // Optional pointer
};

pub fn main() void {
    var node3 = Node{ .value = 30, .next = null };
    var node2 = Node{ .value = 20, .next = &node3 };
    var node1 = Node{ .value = 10, .next = &node2 };

    // Traverse linked list
    var current: ?*Node = &node1;
    while (current) |node| {
        std.debug.print("Value: {}\n", .{node.value});
        current = node.next;
    }
}
```

**출력:**
```
Value: 10
Value: 20
Value: 30
```


---

## 10. 포인터


Zig는 다양한 사용 사례를 위한 여러 포인터 타입을 가지고 있습니다.

## 포인터 타입

```
┌─────────────────────────────────────────┐
│  *T       - 단일 항목 포인터            │
│  [*]T     - 다중 항목 포인터            │
│  *const T - const에 대한 포인터         │
│  ?*T      - 옵셔널 포인터               │
└─────────────────────────────────────────┘
```


```
변수에 대한 포인터:

   ptr          x
┌───────┐   ┌───────┐
│ 주소 ─┼──►│  10   │
└───────┘   └───────┘

*ptr은 값을 얻기 위해 역참조
&x는 x의 주소를 얻음
```


### 예제: pointers_basic

기본 포인터 연산

```zig
const std = @import("std");

pub fn main() void {
    var x: i32 = 10;

    // Get pointer
    const ptr: *i32 = &x;

    // Dereference
    std.debug.print("x = {}\n", .{x});
    std.debug.print("*ptr = {}\n", .{ptr.*});

    // Modify through pointer
    ptr.* = 20;
    std.debug.print("After modification: x = {}\n", .{x});
}
```

**출력:**
```
x = 10
*ptr = 10
After modification: x = 20
```


### 예제: pointer_arithmetic

포인터 산술과 슬라이싱

```zig
const std = @import("std");

pub fn main() void {
    const arr = [_]i32{ 10, 20, 30, 40, 50 };

    // Many-item pointer
    const ptr: [*]const i32 = &arr;

    // Access by index
    std.debug.print("ptr[0] = {}\n", .{ptr[0]});
    std.debug.print("ptr[2] = {}\n", .{ptr[2]});

    // Slice from pointer
    const slice = ptr[1..4];
    for (slice) |val| {
        std.debug.print("slice val: {}\n", .{val});
    }
}
```

**출력:**
```
ptr[0] = 10
ptr[2] = 30
slice val: 20
slice val: 30
slice val: 40
```


---

## 11. 메모리 할당


Zig는 메모리 할당에 대한 명시적 제어를 제공합니다.

## 할당자 인터페이스

```
┌─────────────────────────────────────────┐
│  const allocator = std.heap.page_allocator;│
│  const ptr = try allocator.create(T);   │
│  defer allocator.destroy(ptr);          │
└─────────────────────────────────────────┘
```

## 일반적인 할당자
- `page_allocator`: OS 페이지 할당
- `GeneralPurposeAllocator`: 디버그 친화적
- `ArenaAllocator`: 대량 해제


```
메모리 관리:

스택 (자동):
┌─────────────────┐
│  지역 변수      │ <- 빠름, 자동 정리
└─────────────────┘

힙 (수동):
┌─────────────────┐
│ allocator.alloc │ <- 명시적 할당
│ allocator.free  │ <- 명시적 해제
└─────────────────┘

정리를 위해 defer 사용!
```


### 예제: allocation

동적 메모리 할당

```zig
const std = @import("std");

pub fn main() !void {
    // Use general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Allocate array
    const arr = try allocator.alloc(i32, 5);
    defer allocator.free(arr);

    // Initialize
    for (arr, 0..) |*item, i| {
        item.* = @intCast(i * 10);
    }

    // Print
    for (arr) |val| {
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});
}
```

**출력:**
```
0 10 20 30 40
```


### 예제: arraylist

동적 배열을 위한 ArrayList 사용

```zig
const std = @import("std");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // ArrayList - dynamic array
    var list: std.ArrayListUnmanaged(i32) = .empty;
    defer list.deinit(allocator);

    // Add elements
    try list.append(allocator, 10);
    try list.append(allocator, 20);
    try list.append(allocator, 30);

    std.debug.print("List: ", .{});
    for (list.items) |val| {
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});
    std.debug.print("Length: {}\n", .{list.items.len});
}
```

**출력:**
```
List: 10 20 30 
Length: 3
```


---

## 12. 컴파일 타임 실행


Zig는 `comptime`을 사용하여 컴파일 타임에 코드를 실행할 수 있습니다.

## Comptime 기능

```
┌─────────────────────────────────────────┐
│  comptime {                             │
│      // 컴파일 타임에 실행              │
│  }                                      │
│                                         │
│  fn generic(comptime T: type) type {    │
│      // 타입 수준 계산                  │
│  }                                      │
└─────────────────────────────────────────┘
```


```
컴파일 타임 vs 런타임:

┌──────────────────┐     ┌──────────────────┐
│   컴파일 타임    │     │     런타임       │
├──────────────────┤     ├──────────────────┤
│ comptime 블록    │     │ 일반 코드        │
│ 타입 계산        │     │ 사용자 입력      │
│ 제네릭 매개변수  │     │ 동적 데이터      │
│ 상수 폴딩        │     │ 힙 할당          │
└──────────────────┘     └──────────────────┘
        │                        │
        └────────┬───────────────┘
                 │
            최종 바이너리
```


### 예제: comptime_basic

컴파일 타임 계산

```zig
const std = @import("std");

// Compile-time function
fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Const values with comptime-known args are computed at compile time
const factorial_10 = factorial(10);

pub fn main() void {
    // This value is computed at compile time!
    std.debug.print("10! = {}\n", .{factorial_10});

    // Comptime type introspection
    const T = i32;
    std.debug.print("Size of i32: {} bytes\n", .{@sizeOf(T)});

    // Inline comptime block
    const squares = comptime blk: {
        var arr: [5]i32 = undefined;
        for (0..5) |i| {
            arr[i] = @as(i32, @intCast(i)) * @as(i32, @intCast(i));
        }
        break :blk arr;
    };
    std.debug.print("Squares: ", .{});
    for (squares) |s| {
        std.debug.print("{} ", .{s});
    }
    std.debug.print("\n", .{});
}
```

**출력:**
```
10! = 3628800
Size of i32: 4 bytes
Squares: 0 1 4 9 16
```


### 예제: generic_function

comptime을 사용한 제네릭 함수와 타입

```zig
const std = @import("std");

// Generic max function using comptime
fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

// Generic struct
fn Pair(comptime T: type) type {
    return struct {
        first: T,
        second: T,

        fn swap(self: *@This()) void {
            const tmp = self.first;
            self.first = self.second;
            self.second = tmp;
        }
    };
}

pub fn main() void {
    // Works with any comparable type
    std.debug.print("max(3, 7) = {}\n", .{max(i32, 3, 7)});
    std.debug.print("max(3.5, 2.1) = {d:.1}\n", .{max(f64, 3.5, 2.1)});

    // Generic struct usage
    var pair = Pair(i32){ .first = 10, .second = 20 };
    std.debug.print("Before swap: {}, {}\n", .{ pair.first, pair.second });
    pair.swap();
    std.debug.print("After swap: {}, {}\n", .{ pair.first, pair.second });
}
```

**출력:**
```
max(3, 7) = 7
max(3.5, 2.1) = 3.5
Before swap: 10, 20
After swap: 20, 10
```


---

## 13. 테스팅


Zig는 내장 테스팅 지원을 가지고 있습니다.

## 테스트 문법

```
┌─────────────────────────────────────────┐
│  test "설명" {                          │
│      try std.testing.expect(true);      │
│      try std.testing.expectEqual(1, 1); │
│  }                                      │
└─────────────────────────────────────────┘
```

테스트 실행: `zig test file.zig`


```
테스트 워크플로우:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 테스트 작성 │ -> │  zig test   │ -> │    결과     │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                 통과          실패
                   │             │
                  [OK]      [에러 메시지]
```


### 예제: testing

테스트 작성 및 실행

```zig
const std = @import("std");

fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

test "add function" {
    try std.testing.expectEqual(add(2, 3), 5);
    try std.testing.expectEqual(add(-1, 1), 0);
}

test "divide function" {
    const result = try divide(10, 2);
    try std.testing.expectEqual(result, 5);
}

test "divide by zero returns error" {
    const result = divide(10, 0);
    try std.testing.expectError(error.DivisionByZero, result);
}

pub fn main() void {
    std.debug.print("Run 'zig test' to execute tests\n", .{});
    std.debug.print("add(2,3) = {}\n", .{add(2, 3)});
}
```

**출력:**
```
Run 'zig test' to execute tests
add(2,3) = 5
```


---

## 14. 빌드 시스템


Zig는 `build.zig`를 사용하는 강력한 내장 빌드 시스템을 가지고 있습니다.

## 기본 build.zig

```
┌─────────────────────────────────────────┐
│  zig init     - 새 프로젝트 생성        │
│  zig build    - 프로젝트 빌드           │
│  zig build run - 빌드 후 실행           │
└─────────────────────────────────────────┘
```


```
프로젝트 구조:

my_project/
├── build.zig       <- 빌드 설정
├── build.zig.zon   <- 의존성
└── src/
    ├── main.zig    <- 진입점
    └── lib.zig     <- 라이브러리 코드

빌드 프로세스:
┌────────────┐   ┌────────────┐   ┌────────────┐
│ build.zig  │ → │ zig build  │ → │ zig-out/   │
└────────────┘   └────────────┘   └────────────┘
```


### 예제: build_example

프로젝트 구조와 빌드 명령어

```zig
const std = @import("std");

// Example of a library function
pub fn greet(name: []const u8) void {
    std.debug.print("Hello, {s}!\n", .{name});
}

// Example of configuration
pub const Config = struct {
    debug: bool = false,
    optimization: enum { none, speed, size } = .none,
};

pub fn main() void {
    std.debug.print("Build System Demo\n", .{});
    std.debug.print("================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Commands:\n", .{});
    std.debug.print("  zig init        - Create new project\n", .{});
    std.debug.print("  zig build       - Build project\n", .{});
    std.debug.print("  zig build run   - Build and run\n", .{});
    std.debug.print("  zig build test  - Run tests\n", .{});
    std.debug.print("\n", .{});
    greet("World");
}
```

**출력:**
```
Build System Demo
================

Commands:
  zig init        - Create new project
  zig build       - Build project
  zig build run   - Build and run
  zig build test  - Run tests

Hello, World!
```


---

## 빠른 참조


### 일반적인 명령어
```
zig run file.zig     # 컴파일 후 실행
zig build-exe file.zig  # 실행 파일로 컴파일
zig test file.zig    # 테스트 실행
zig fmt file.zig     # 코드 포맷팅
zig init             # 새 프로젝트 초기화
zig build            # 프로젝트 빌드
```

### 유용한 내장 함수
| 내장 함수 | 설명 |
|----------|-------------|
| @import | 모듈 가져오기 |
| @intCast | 정수 타입 캐스트 |
| @floatFromInt | 정수를 부동소수점으로 변환 |
| @truncate | 더 작은 타입으로 자르기 |
| @sizeOf | 타입의 크기 얻기 |
| @TypeOf | 표현식의 타입 얻기 |

### 포맷 지정자
| 지정자 | 설명 |
|-----------|-------------|
| {} | 기본값 |
| {s} | 문자열 |
| {c} | 문자 |
| {d} | 10진수 |
| {x} | 16진수 |
| {b} | 2진수 |
