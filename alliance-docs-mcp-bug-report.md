# Bug Report: alliance-docs MCP Server - Empty Search Results

**Status**: ✅ RESOLVED (2025-01-15)

## Summary

The `alliance-docs` MCP server's search functions were returning empty results for all queries, despite the server appearing to be connected and functional (category listing worked correctly). **This issue has been resolved** - search functionality is now working correctly.

## Environment

- **MCP Server**: `alliance-docs`
- **Date**: 2025-01-15
- **Client**: Cursor IDE with MCP integration
- **Python Environment**: Python 3.11.13 (via uv)

## Issue Description

When attempting to search the alliance-docs documentation using the `mcp_alliance-docs_search_docs` tool, all queries return empty results. The tool executes without errors, but returns no content regardless of the search query.

### Observed Behavior

1. **Search queries return empty**: All calls to `mcp_alliance-docs_search_docs` return empty results
2. **Category listing works**: `mcp_alliance-docs_list_categories` successfully returns a list of categories
3. **No error messages**: The tool executes without throwing errors or warnings
4. **Multiple query types tested**: Various search terms were attempted with no results

## Steps to Reproduce

1. Attempt to search alliance-docs using `mcp_alliance-docs_search_docs` with any query
2. Query executes successfully (no errors)
3. Empty results are returned

### Queries Tested

The following queries all returned empty results:

- `"DRAC nibi compute cluster"`
- `"how to run jobs on compute cluster"`
- `"nibi"`
- `"DRAC"`
- `"compute cluster GPU"`
- `"cluster job scheduler slurm"`
- `"HPC high performance computing"`

### Successful Operation

The following operation **does** work correctly:

- `mcp_alliance-docs_list_categories()` → Returns: `["Cybersecurity","General","Getting Started","META-Farm","Migration2016","Technical Reference","User Guide","Using Nix"]`

## Expected Behavior

Search queries should return relevant documentation pages matching the search terms, similar to how the category listing successfully returns available categories.

## Actual Behavior (Before Fix)

All search queries returned empty results, making it impossible to find documentation pages using the search functionality.

## Resolved Behavior (After Fix)

All search queries now return relevant results. Tests performed on 2025-01-15 confirm:
- ✅ "DRAC nibi compute cluster" → Returns 21 relevant results
- ✅ "nibi" → Returns 21 results about the Nibi cluster
- ✅ "compute cluster GPU" → Returns many GPU-related results  
- ✅ "how to run jobs on compute cluster" → Returns "Running jobs/en" and related pages
- ✅ "HPC high performance computing" → Returns many HPC-related results

## Impact

- **Severity**: High
- **User Impact**: Cannot search for documentation using the MCP server
- **Workaround**: None identified - category listing works but doesn't provide search functionality

## Additional Information

### Tool Function Signature

The tool appears to have the following interface:

```python
mcp_alliance-docs_search_docs(
    query: str,
    category: Optional[str] = None,
    limit: int = 20,
    search_content: bool = True,
    fuzzy: bool = False
)
```

### Observations

1. The server is connected (category listing confirms this)
2. The search tool is callable (no execution errors)
3. The issue appears specific to search functionality, not general server connectivity
4. No error messages or logs indicate what might be causing the empty results

## Possible Causes

1. **Index not built**: Search index may not be initialized or populated
2. **Query parsing issue**: Search queries may not be parsed correctly
3. **Database/backend issue**: Documentation content may not be indexed or accessible
4. **Configuration issue**: Search functionality may be disabled or misconfigured
5. **Empty database**: Documentation content may not be loaded into the server

## Recommended Fixes

1. Verify that the documentation content is loaded and indexed
2. Check search index initialization on server startup
3. Add error logging to search function to identify why no results are returned
4. Verify that the search backend (database/index) is properly configured
5. Test with very simple queries (single words) to isolate the issue
6. Check if there are any search-related configuration parameters that need to be set

## Related Tools

The following tools were tested:

- ✅ `mcp_alliance-docs_list_categories` - Works correctly
- ❌ `mcp_alliance-docs_search_docs` - Returns empty results
- ❓ `mcp_alliance-docs_get_page_by_title` - Not tested
- ❓ `mcp_alliance-docs_get_page_content` - Not tested
- ❓ `mcp_alliance-docs_list_all_pages` - Not tested

## Requested Actions

1. Investigate why search queries return empty results
2. Add debug logging to identify the root cause
3. Verify documentation content is loaded and searchable
4. Test search functionality with various query types
5. Provide fix or workaround
